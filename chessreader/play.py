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
from .display import (GREEN, RED, YELLOW, draw_change, draw_choices, draw_eval_bar, draw_grid, draw_hud,
                      draw_mask_outline, draw_occupancy, draw_quad, highlight_move, put_text, square_px)
from .engine import EngineWorker, find_stockfish
from .geometry import BoardCamera
from .health import compare_occupancy
from .session import GameSession, Phase, SessionHooks
from .settings import PROJECT_DIR, Settings, resolve_path
from .stream import FrameGrabber

logger = logging.getLogger(__name__)

CAMERA_WINDOW = "Camera"
BOARD_WINDOW = "Board"
KEYS = "S start  E end  G resign  A abort  N new  R reset  Z undo  X ignore  T type move  C recalibrate  Q quit"
SETUP_MESSAGE = "Set up the starting position, then press S."


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


PANEL_HEIGHT = 214  # eval line, moves, opening, status, 2 engine lines
_opening_cache: dict = {}


def opening_book(settings: Settings):
    """The opening book (loaded once), or None when it's off or can't be loaded."""
    o = settings.openings
    if not o.enabled:
        return None
    key = ("book", o.book or "")
    if key not in _opening_cache:
        try:
            from .openings import book_from

            _opening_cache[key] = book_from(resolve_path(o.book) if o.book else None)
        except Exception as exc:  # a broken book must never stop the game
            logger.warning("Opening book not loaded (%s); openings off", exc)
            _opening_cache[key] = None
    return _opening_cache[key]


def opening_text(settings: Settings, board: chess.Board, reveal: bool) -> str | None:
    """'C60 Ruy Lopez   theory: a6 Nf6 d6' or '... out of book at move 6', for the Board panel."""
    book = opening_book(settings)
    if book is None:
        return None
    key = ("text", board.fen(), len(board.move_stack), reveal)
    if key not in _opening_cache:
        info = book.lookup(board)
        if info.name is None and not info.in_book:
            text = None
        else:
            text = f"{info.eco or ''} {info.name or ''}".strip()
            n = settings.openings.show_book_moves
            if info.in_book and reveal and n > 0:
                moves = book.book_san(board, n)
                if moves:
                    text += "   theory: " + " ".join(moves)
            elif info.left_book_ply:
                text += f"   (out of book at move {(info.left_book_ply + 1) // 2})"
        if len(_opening_cache) > 500:
            _opening_cache.clear()
        _opening_cache[key] = text
    return _opening_cache[key]


def render_board(session: GameSession, frame: np.ndarray, settings: Settings) -> np.ndarray:
    try:
        warp = session.cal.warp(frame)
    except CalibrationError as exc:
        warp = np.zeros((session.cal.warp_size, session.cal.warp_size, 3), np.uint8)
        draw_hud(warp, [str(exc)[:60]], color=RED)
        return warp
    draw_grid(warp, settings.view.show_labels and session.phase != Phase.WAIT_START)
    board = session.board
    # last move as a tinted from/to square pair; while a question is open only the candidates show
    if board.move_stack and not session.choices:
        highlight_move(warp, board.peek())
    draw_occupancy(warp, session.occupancy)
    for sq in session.desync:
        x, y = square_px(chess.parse_square(sq), warp.shape[0])
        half = warp.shape[0] // 16
        cv2.rectangle(warp, (x - half, y - half), (x + half, y + half), RED, 3)
    if session.choices:
        draw_choices(warp, session.choices, YELLOW)

    an = session.analysis
    # best moves stay hidden during play in training/review mode (engine.show_best_moves: false)
    reveal = settings.engine.show_best_moves or session.phase == Phase.ENDED
    panel = np.full((PANEL_HEIGHT, warp.shape[1], 3), 25, np.uint8)

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
    lines = [" ".join(sans[-12:]) or "(no moves yet)"]
    opening = opening_text(settings, board, reveal) if session.record else None
    lines.append(opening[:80] if opening else "")
    lines.append(f"{session.phase.value}   {tier}")
    if an and reveal:
        lines += [f"{ev}  {line}" for ev, line in an.lines[:2]]
    draw_hud(panel, lines, origin=(12, 92))
    out = np.vstack([warp, panel])
    return draw_eval_bar(out, an.eval_cp if an else None, an.mate if an else None)


def looks_like_start(session: GameSession, frame: np.ndarray) -> bool | None:
    """Does the frame show the starting position rather than the game's current one? None = can't tell.

    Uses the square looks the session learned at its start. A low camera sees a few squares wrong
    whatever the position (pieces lean over their neighbours), so this compares how many squares
    disagree with the start against how many disagree with the position the game is at.
    """
    if session.cam is None or not session.occ_reader.templates:
        return None
    start, now = chess.Board(), session.board
    if now.board_fen() == start.board_fen():
        return None
    try:
        warped = session.cal.warp(frame)
    except CalibrationError:
        return None
    min_conf = session.s.health.sync_min_confidence
    vs_start = compare_occupancy(session.occ_reader.read(warped, session.cam, start), start, min_conf)
    vs_now = compare_occupancy(session.occ_reader.read(warped, session.cam, now), now, min_conf)
    if not vs_start.checked:
        return None
    logger.debug("Reset: %d squares off the start position, %d off the game", len(vs_start.mismatches),
                 len(vs_now.mismatches))
    return len(vs_start.mismatches) < len(vs_now.mismatches)


def reset_game(session: GameSession, frame: np.ndarray | None, new_session) -> GameSession:
    """R: end the current game (its PGN stays, result *) and start a fresh one, right away if the start position is up.

    Works in any phase. new_session(cal) builds the next GameSession, as N does.
    """
    old_record = session.record
    if old_record and session.phase != Phase.ENDED:
        session.end("unknown")
    at_start = looks_like_start(session, frame) if frame is not None else False
    logger.info("Game reset by the player (%d plies dropped%s)", len(session.moves),
                f", PGN kept: {old_record.path}" if old_record else "")
    session.close()
    fresh = new_session(session.cal)
    fresh.message = SETUP_MESSAGE
    if frame is not None and at_start is not False:
        error = fresh.start_game(frame)
        if error:
            fresh.message = f"{error} Then press S."
    if fresh.phase == Phase.PLAYING:
        fresh.message = "Game reset. " + fresh.message
    return fresh


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
    opening_book(settings)  # load the book now, not on the first move
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
    session.message = SETUP_MESSAGE
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
            elif ch == "r":
                session = reset_game(session, frame, new_session)
                cache.clear()
            elif ch == "g" and session.record:
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
                session.message = SETUP_MESSAGE
    finally:
        if session.record and session.phase not in (Phase.ENDED, Phase.WAIT_START) and session.moves:
            session.end("unknown")
        session.close()
        services.close()
        cv2.destroyAllWindows()
    return 0
