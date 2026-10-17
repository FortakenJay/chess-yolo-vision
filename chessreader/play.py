"""Live play: camera -> GameSession -> on-screen windows (board, eval, Stockfish lines) and a PGN file."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import chess
import cv2
import numpy as np
from dotenv import load_dotenv

from .calibration import Calibration, CalibrationError, auto_calibrate, run_calibration
from .confirm import LlmJudge
from .display import (GREEN, RED, YELLOW, draw_change, draw_eval_bar, draw_grid, draw_hud, draw_mask_outline,
                      draw_move, draw_occupancy, draw_quad, put_text, square_px)
from .engine import EngineWorker, find_stockfish
from .geometry import BoardCamera
from .session import GameSession, Phase, SessionHooks
from .settings import PROJECT_DIR, Settings, resolve_path
from .stream import FrameGrabber

logger = logging.getLogger(__name__)

CAMERA_WINDOW = "Camera"
BOARD_WINDOW = "Board"
KEYS = "S start  E end  R resign  A abort  N new  Z undo  X ignore  T type move  C recalibrate  Q quit"


def make_promotion_chooser(settings: Settings, session_ref: list):
    """Look at the promotion square with the YOLO model; None means 'use a queen'."""
    detector = []

    def choose(frame: np.ndarray, cam: BoardCamera, square: int, color: bool) -> int | None:
        if not settings.pieces.use_for_promotion:
            return None
        if not detector:
            from .reader import PieceDetector

            detector.append(PieceDetector(resolve_path(settings.pieces.weights), settings.pieces.conf))
        session: GameSession = session_ref[0]
        sq_px = session.cal.square_px
        best: tuple[float, int] | None = None
        types = {"pawn": chess.PAWN, "knight": chess.KNIGHT, "bishop": chess.BISHOP,
                 "rook": chess.ROOK, "queen": chess.QUEEN, "king": chess.KING}
        for det in detector[0].detect(frame):
            parts = det.label.split("-")
            piece_color = None if len(parts) == 1 else parts[0] == "white"
            piece_type = types.get(parts[-1])
            if piece_type is None or (piece_color is not None and piece_color != color):
                continue
            x, y = cv2.perspectiveTransform(np.array([[det.base_point]], np.float32), session.cal.homography)[0, 0]
            col, row = int(x // sq_px), int(y // sq_px)
            if 0 <= col < 8 and 0 <= row < 8 and chess.square(col, 7 - row) == square:
                if best is None or det.confidence > best[0]:
                    best = (det.confidence, piece_type)
        return best[1] if best else None

    return choose


class Services:
    """Stockfish analysis (optional); results are shown in the Board window."""

    def __init__(self, settings: Settings) -> None:
        self.engine = None
        e = settings.engine
        if not e.enabled:
            return
        path = find_stockfish(e.path)
        if path is None:
            logger.warning("Stockfish not found (engine.path is %r, none on PATH either); analysis off. "
                           "Install it (macOS: brew install stockfish) or set engine.path", e.path)
            return
        try:
            self.engine = EngineWorker(path, e.think_time_s, e.multipv).start()
        except FileNotFoundError as exc:
            logger.warning("%s; analysis off", exc)

    def hooks(self, session_ref: list, settings: Settings) -> SessionHooks:
        hooks = SessionHooks(choose_promotion=make_promotion_chooser(settings, session_ref))
        if self.engine:
            hooks.analyse = self.engine.submit
            self.engine.listeners.append(lambda a: setattr(session_ref[0], "analysis", a))
        return hooks

    def close(self) -> None:
        if self.engine:
            self.engine.stop()


def render_camera(session: GameSession, frame: np.ndarray, grabber: FrameGrabber, cache: dict) -> np.ndarray:
    vis = frame.copy()
    if session.matcher is not None:
        roi = session.matcher.roi
        if session.last_change is not None and session.phase != Phase.WAIT_START:
            draw_change(vis, session.last_change, roi)
        result = session.last_result
        if result is not None and result.best is not None and session.phase in (Phase.VOTING, Phase.ASK_USER, Phase.WAIT_LLM):
            key = (id(result), len(session.board.move_stack))
            if cache.get("key") != key:
                before = session.matcher.render(session.board)
                cache["key"] = key
                cache["mask"] = session.matcher.predicted_mask(before, session.board, result.best.move)
            draw_mask_outline(vis, cache["mask"], roi)
    draw_quad(vis, session.cal.outer_corners, GREEN)
    status = "live" if grabber.connected else "RECONNECTING"
    gate = session.gate_state.value if session.gate_state else "-"
    lines = [f"{grabber.fps:.1f} fps {status} | board {gate} | {session.phase.value}"]
    if session.last_result is not None and session.last_result.best is not None:
        r = session.last_result
        lines.append(f"best {r.best.move.uci()} score {r.best.score:.0f} margin {r.margin:.0f} "
                     f"unexplained {r.unexplained:.2f}")
    if session.message:
        lines.append(session.message[:90])
    draw_hud(vis, lines)
    return vis


def render_board(session: GameSession, frame: np.ndarray, settings: Settings) -> np.ndarray:
    try:
        warp = session.cal.warp(frame)
    except CalibrationError as exc:
        warp = np.zeros((session.cal.warp_size, session.cal.warp_size, 3), np.uint8)
        draw_hud(warp, [str(exc)[:60]], color=RED)
        return warp
    draw_grid(warp, settings.view.show_labels and session.phase != Phase.WAIT_START)
    draw_occupancy(warp, session.occupancy)
    board = session.board
    if board.move_stack:
        draw_move(warp, board.peek())
    for i, move in enumerate(session.choices):
        draw_move(warp, move, YELLOW)
        x, y = square_px(move.to_square, warp.shape[0])
        put_text(warp, str(i + 1), (x - 8, y + 10), scale=1.2, color=YELLOW, thickness=2)
    for sq in session.desync:
        x, y = square_px(chess.parse_square(sq), warp.shape[0])
        half = warp.shape[0] // 16
        cv2.rectangle(warp, (x - half, y - half), (x + half, y + half), RED, 3)

    an = session.analysis
    # best moves stay hidden during play in training/review mode (engine.show_best_moves: false)
    reveal = settings.engine.show_best_moves or session.phase == Phase.ENDED
    panel = np.full((190, warp.shape[1], 3), 25, np.uint8)

    # big eval + best move, like the e-ink display would have shown
    eval_text = an.eval_text if an else "..."
    put_text(panel, eval_text, (12, 52), scale=1.6, thickness=3)
    if an and reveal and an.best_move:
        put_text(panel, f"best: {an.best_move}", (190, 52), scale=1.1, color=YELLOW, thickness=2)
    if an and an.wdl:
        w, d, l = (v / 10 for v in an.wdl)
        put_text(panel, f"W {w:.0f}%  D {d:.0f}%  L {l:.0f}%", (500, 50), scale=0.7)

    sans = []
    replay = chess.Board(session.record.starting_fen) if session.record else chess.Board()
    for move in board.move_stack:
        prefix = f"{replay.fullmove_number}." if replay.turn else ""
        sans.append(prefix + replay.san(move))
        replay.push(move)
    tier = f"read by tier {session.last_tier}" if session.last_tier else ""
    lines = [" ".join(sans[-12:]) or "(no moves yet)", f"{session.phase.value}   {tier}"]
    if an and reveal:
        lines += [f"{ev}  {line}" for ev, line in an.lines[:2]]
    draw_hud(panel, lines, origin=(12, 92))
    out = np.vstack([warp, panel])
    return draw_eval_bar(out, an.eval_cp if an else None, an.mate if an else None)


def _prompt(text: str) -> str:
    try:
        return input(text)
    except EOFError:
        return ""


def run_play(settings: Settings, grabber: FrameGrabber, auto_start: bool = False) -> int:
    load_dotenv(PROJECT_DIR / ".env")
    cal_dir = resolve_path(settings.calibration.dir)
    try:
        cal = Calibration.load(cal_dir)
    except FileNotFoundError:
        cal = None
    first = None
    cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
    while first is None:  # wait for the camera instead of giving up
        first = grabber.wait_for_frame(timeout=0.5)
        blank = np.zeros((300, 640, 3), np.uint8)
        draw_hud(blank, ["Waiting for the camera...", "Is this PC on ChessCam? Q to quit"])
        cv2.imshow(CAMERA_WINDOW, blank)
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            return 1
    h, w = first.image.shape[:2]
    if cal is not None and tuple(cal.frame_size) != (w, h):
        logger.info("Saved calibration is for %s, camera sends %dx%d; recalibrating", cal.frame_size, w, h)
        cal = None
    if cal is None and auto_start:
        cal = auto_calibrate(grabber, settings.calibration)
    if cal is None:
        cal = run_calibration(grabber, settings.calibration, settings.view.show_labels)
        if cal is None:
            return 1

    services = Services(settings)
    llm = LlmJudge(settings.llm.model, settings.llm.enabled) if settings.llm.enabled else None
    sessions_root = resolve_path(settings.storage.sessions_dir)

    def new_session(c: Calibration) -> GameSession:
        ref: list = []
        directory = sessions_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        s = GameSession(settings, c, services.hooks(ref, settings), llm, directory)
        ref.append(s)
        logger.info("Session folder: %s", directory)
        return s

    session = new_session(cal)
    session.message = "Set up the starting position, then press S."
    cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(BOARD_WINDOW, cv2.WINDOW_NORMAL)
    print(f"\nKeys (in a camera window): {KEYS}\n")
    last_index, frame, cache = -1, first.image, {}
    try:
        while True:
            latest = grabber.wait_for_frame(last_index, timeout=0.1)
            if latest is not None:
                last_index, frame = latest.index, latest.image
                if auto_start and session.phase == Phase.WAIT_START:
                    error = session.start_game(frame)
                    if error:
                        session.message = error
                session.process(frame, latest.timestamp)
            cv2.imshow(CAMERA_WINDOW, render_camera(session, frame, grabber, cache))
            cv2.imshow(BOARD_WINDOW, render_board(session, frame, settings))
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                if cv2.getWindowProperty(BOARD_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            ch = chr(key).lower()
            if ch == "q":
                break
            elif ch == "s" and session.phase == Phase.WAIT_START:
                error = session.start_game(frame)
                if error:
                    session.message = error
            elif ch in "123":
                session.choose(int(ch) - 1)
            elif ch == "x":
                session.ignore_change()
            elif ch == "z":
                session.undo()
            elif ch == "t" and session.record:
                error = session.type_move(_prompt("Move (SAN like Nf3 or UCI like g1f3): "))
                if error:
                    session.message = error
            elif ch == "c":
                new = run_calibration(grabber, settings.calibration, settings.view.show_labels)
                if new is not None:
                    if session.record and session.phase != Phase.ENDED:
                        error = session.recalibrated(new, grabber.latest().image)
                        session.message = error or session.message
                    else:
                        session.cal = new
            elif ch == "e" and session.record:
                answer = _prompt("Result? w = white won, b = black won, d = draw, u = unknown: ").strip().lower()
                session.end({"w": "white", "b": "black", "d": "draw"}.get(answer[:1], "unknown"))
            elif ch == "r" and session.record:
                answer = _prompt("Who resigned? w / b: ").strip().lower()
                if answer[:1] in ("w", "b"):
                    session.end("black" if answer[:1] == "w" else "white")
            elif ch == "a" and session.record:
                session.phase = Phase.ENDED
                session.message = "Game aborted (PGN kept)."
                logger.info("Game aborted")
            elif ch == "n" and session.phase == Phase.ENDED:
                session.close()
                session = new_session(session.cal)
                session.message = "Set up the starting position, then press S."
    finally:
        if session.record and session.phase not in (Phase.ENDED, Phase.WAIT_START) and session.moves:
            session.end("unknown")
        session.close()
        services.close()
        cv2.destroyAllWindows()
    return 0
