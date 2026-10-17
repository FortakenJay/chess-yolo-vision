"""One game: frames in, accepted moves out. No UI here, so live play and replay share it.

Per frame: the motion gate decides when the board is still. On each new still board the
change since the last accepted position is matched against every legal move, then the
confirmation tiers decide (auto, vote over re-reads, LLM, or ask the player).
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import chess
import cv2
import numpy as np

from .calibration import Calibration, edge_map
from .confirm import ConfirmSettings, LlmJudge, Tier, average_results, tier1_ok, tier2_ok
from .engine import Analysis
from .gating import GateState, MotionGate
from .geometry import BoardCamera, orient_from_start, square_center
from .health import DriftMonitor, SyncMonitor, compare_occupancy
from .matcher import MatchResult, MoveMatcher
from .reader import ChangeReader, OccupancyReader, Prepared
from .settings import Settings
from .storage import GameRecord

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    WAIT_START = "set up the start position, press S"
    PLAYING = "playing"
    VOTING = "re-reading"
    WAIT_LLM = "asking the LLM"
    ASK_USER = "which move? press 1-3"
    PAUSED = "paused"
    ENDED = "game over"


@dataclass
class MoveLog:
    ply: int
    uci: str
    san: str
    tier: int
    margin: float
    score: float
    unexplained: float
    timestamp: float


@dataclass
class SessionHooks:
    """Optional outside services; each may be None."""
    on_move: list = field(default_factory=list)          # callables(record: GameRecord)
    on_game_start: list = field(default_factory=list)    # callables(record: GameRecord)
    on_game_end: list = field(default_factory=list)      # callables(record: GameRecord)
    analyse: object = None                               # callable(board) -> None (async)
    choose_promotion: object = None                      # callable(frame, cam, square, color) -> piece_type | None
    ask_user: object = None                              # replay: callable(board, candidates) -> Move | None


class GameSession:
    def __init__(self, settings: Settings, cal: Calibration, hooks: SessionHooks | None = None,
                 llm: LlmJudge | None = None, session_dir: Path | None = None) -> None:
        self.s = settings
        self.cal = cal
        self.hooks = hooks or SessionHooks()
        self.llm = llm
        self.confirm = ConfirmSettings(**settings.confirm.model_dump())
        self.phase = Phase.WAIT_START
        self.message = ""
        self.record: GameRecord | None = None
        self.moves: list[MoveLog] = []
        self.last_result: MatchResult | None = None
        self.last_change: np.ndarray | None = None
        self.choices: list[chess.Move] = []
        self.occupancy: np.ndarray | None = None
        self.desync: list[str] = []
        self.analysis: Analysis | None = None
        self.last_tier: int | None = None

        self.cam: BoardCamera | None = None
        self.matcher: MoveMatcher | None = None
        self.reader: ChangeReader | None = None
        self.gate: MotionGate | None = None
        self.occ_reader = OccupancyReader()
        self.sync = SyncMonitor()
        self.drift: DriftMonitor | None = None

        self._ref: Prepared | None = None
        self._ref_image: np.ndarray | None = None
        self._ref_stack: list[tuple[Prepared, np.ndarray]] = []
        self._votes: list[MatchResult] = []
        self._last_frame: np.ndarray | None = None
        self._last_drift_t = 0.0
        self._moves_since_sync = 0
        self._llm_thread: threading.Thread | None = None
        self._llm_answer: list = []
        self._ask_frame: np.ndarray | None = None
        self._vote_frame: np.ndarray | None = None

        self.session_dir = session_dir
        self._frames_csv = None
        self._moves_csv = None
        if session_dir is not None:
            (session_dir / "frames").mkdir(parents=True, exist_ok=True)
            self._frames_csv = open(session_dir / "frames.csv", "a", newline="", encoding="utf-8")
            self._moves_csv = open(session_dir / "moves.csv", "a", newline="", encoding="utf-8")
            csv.writer(self._moves_csv).writerow(["ply", "uci", "san", "tier", "margin", "score", "unexplained", "t"])
        self._frame_no = 0

    # ------------------------------------------------------------------ properties

    @property
    def board(self) -> chess.Board:
        return self.record.board if self.record else chess.Board()

    @property
    def gate_state(self) -> GateState | None:
        return self.gate.state if self.gate else None

    # ------------------------------------------------------------------ setup

    def _build(self, cal: Calibration) -> None:
        r = self.s.reader
        self.cal = cal
        self.cam = BoardCamera.from_calibration(cal, r.height_scale)
        roi = self.cam.roi()
        self.matcher = MoveMatcher(self.cam, roi, r.scale, r.q_in, r.q_out)
        self.reader = ChangeReader(roi, r.scale, t_low=r.t_low, t_high=r.t_high)
        x0, y0, x1, y1 = roi
        region_full = cv2.resize(self.matcher.region.astype(np.uint8), (x1 - x0, y1 - y0),
                                 interpolation=cv2.INTER_NEAREST)
        g = self.s.gate
        self.gate = MotionGate(region_full, g.pixel_threshold, g.motion_fraction, g.stable_time_s, g.min_stable_frames)
        self._region_px = float(self.matcher.region.sum()) * self.matcher.px

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.matcher.roi
        return frame[y0:y1, x0:x1]

    def start_game(self, frame: np.ndarray, board: chess.Board | None = None) -> str | None:
        """Orient the board from the position on it and start recording. Returns an error or None."""
        position = board or chess.Board()
        try:
            cal, cam, contrast = orient_from_start(self.cal, frame, self.s.reader.height_scale, position)
        except ValueError as exc:
            return str(exc)
        if contrast < 25:
            return f"Can't see the pieces clearly (contrast {contrast:.0f}). Is the start position set up?"
        self._build(cal)
        if self.session_dir is not None:
            import json
            (self.session_dir / "calibration.json").write_text(json.dumps(cal.to_json(), indent=2), encoding="utf-8")
        self.drift = DriftMonitor(edge_map(cal.warp(frame), self.s.calibration.canny_low, self.s.calibration.canny_high),
                                  self.s.calibration.canny_low, self.s.calibration.canny_high,
                                  threshold=self.s.health.drift_threshold)
        warped = cal.warp(frame)
        self.occ_reader.fit(warped, self.cam, position)
        st = self.s.storage
        from .settings import resolve_path
        self.record = GameRecord(resolve_path(st.exports_dir), st.white, st.black, starting_fen=position.fen())
        self._set_reference(frame)
        self._ref_stack.clear()
        self.moves.clear()
        self.phase = Phase.PLAYING
        self.message = f"Game started (white at {self._white_side()}). Make the first move."
        logger.info("Game started; orientation contrast %.0f; PGN %s", contrast, self.record.path)
        for hook in self.hooks.on_game_start:
            hook(self.record)
        if self.hooks.analyse:
            self.hooks.analyse(self.board)
        return None

    def _white_side(self) -> str:
        """Where white sits in the camera image, in words (for the LLM and the HUD)."""
        x, y = self.cam.project(np.array([[4.0, 8.0, 0.0]]))[0]   # middle of white's back edge
        cx, cy = self.cam.project(np.array([[4.0, 4.0, 0.0]]))[0]
        dx, dy = x - cx, y - cy
        if abs(dx) > abs(dy):
            return "the left of the image" if dx < 0 else "the right of the image"
        return "the top of the image" if dy < 0 else "the bottom of the image"

    def recalibrated(self, cal: Calibration, frame: np.ndarray) -> str | None:
        """Use a new calibration mid-game: orient it from the current position."""
        position = self.board.copy()
        try:
            cal, _, contrast = orient_from_start(cal, frame, self.s.reader.height_scale, position)
        except ValueError as exc:
            return str(exc)
        self._build(cal)
        self.drift.reset(edge_map(cal.warp(frame), self.s.calibration.canny_low, self.s.calibration.canny_high))
        self._set_reference(frame)
        self.phase = Phase.PLAYING
        self.message = "Recalibrated."
        logger.info("Recalibrated mid-game (contrast %.0f)", contrast)
        return None

    # ------------------------------------------------------------------ per frame

    def _set_reference(self, frame: np.ndarray) -> None:
        self._ref = self.reader.prepare(frame)
        self._ref_image = frame.copy()

    def _read(self, frame: np.ndarray) -> MatchResult:
        cur = self.reader.prepare(frame)
        change = self.reader.change_probability(self._ref, cur, self.matcher.region)
        self.last_change = change
        result = self.matcher.match(self.board, change)
        self.last_result = result
        best = result.best
        logger.debug("read: change %.0fpx, best %s %.0f, margin %.0f, unexplained %.2f",
                     result.total_change, best.move.uci() if best else None,
                     best.score if best else 0, result.margin, result.unexplained)
        return result

    def _save_frame(self, frame: np.ndarray, timestamp: float) -> None:
        if self.session_dir is None or not self.s.storage.record_frames:
            return
        name = f"{self._frame_no:06d}.jpg"
        cv2.imwrite(str(self.session_dir / "frames" / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        csv.writer(self._frames_csv).writerow([name, f"{timestamp:.3f}", self.phase.name])
        self._frames_csv.flush()
        self._frame_no += 1

    def process(self, frame: np.ndarray, timestamp: float) -> None:
        self._save_frame(frame, timestamp)
        self._last_frame = frame
        if self.phase in (Phase.WAIT_START, Phase.ENDED) or self.gate is None:
            return
        update = self.gate.update(self._crop(frame), timestamp)

        if self.phase == Phase.PAUSED:
            return
        if self.phase == Phase.WAIT_LLM:
            self._check_llm(frame)
            return
        if self.phase == Phase.ASK_USER:
            if update.became_stable:  # the player may have fixed the board instead of answering
                result = self._read(frame)
                if result.total_change < self.s.reader.min_change_px:
                    self.choices = []
                    self.phase = Phase.PLAYING
                    self.message = "Board matches the last position again."
            return
        if self.phase == Phase.VOTING:
            if update.state == GateState.MOVING:
                # the next move is starting: settle this one with the reads we have
                self._decide_after_vote(self._vote_frame)
                return
            if update.state == GateState.STABLE:
                self._votes.append(self._read(frame))
                self._vote_frame = frame
                if len(self._votes) > self.confirm.vote_frames:
                    self._decide_after_vote(frame)
            return

        # PLAYING
        if update.state == GateState.STABLE and timestamp - self._last_drift_t > self.s.health.drift_every_s:
            self._last_drift_t = timestamp
            if self.drift is not None and self.drift.update(self.cal.warp(frame)):
                self.phase = Phase.PAUSED
                self.message = "The board or camera moved. Press C to recalibrate (pieces can stay)."
                return
        if update.became_stable:
            self._decide(self._read(frame), frame)

    # ------------------------------------------------------------------ decisions

    def _decide(self, result: MatchResult, frame: np.ndarray) -> None:
        r = self.s.reader
        if result.total_change < r.min_change_px:
            return  # same position as before
        if result.total_change > r.occlusion_fraction * self._region_px:
            self.message = "Something is covering the board."
            return
        if result.best is None or result.best.score < self.confirm.min_score:
            if result.total_change < r.unexplained_alarm_px:
                # e.g. lighting / exposure drift: no move explains it, so nothing happened
                logger.debug("ignoring %.0fpx of change that no move explains", result.total_change)
                return
            logger.info("Board changed (%.0fpx) but no legal move fits", result.total_change)
            self._ask_user([c.move for c in result.top(3)], frame)
            self.message = "The board changed but no legal move fits. " + self.message
            return
        if tier1_ok(result, self.confirm):
            self._accept(result.best.move, Tier.AUTO, result, frame)
            return
        self._votes = [result]
        self._vote_frame = frame
        self.phase = Phase.VOTING
        self.message = f"Not sure ({result.best.move.uci()} by {result.margin:.0f}); re-reading..."

    def _decide_after_vote(self, frame: np.ndarray) -> None:
        merged = average_results(self._votes)
        self._votes = []
        self.last_result = merged
        if tier2_ok(merged, self.confirm):
            self._accept(merged.best.move, Tier.VOTE, merged, frame)
            return
        candidates = [c.move for c in merged.top(3)]
        if self.llm is not None and self.llm.available:
            self._start_llm(candidates, frame)
            return
        self._ask_user(candidates, frame)

    def _start_llm(self, candidates: list[chess.Move], frame: np.ndarray) -> None:
        self.phase = Phase.WAIT_LLM
        self.message = "Asking the LLM..."
        self._llm_answer = []
        board, before, after = self.board.copy(), self._ref_image.copy(), frame.copy()
        note = f"White sits at {self._white_side()}."

        def run() -> None:
            self._llm_answer.append(self.llm.choose(board, before, after, candidates, note))

        self._llm_candidates = candidates
        self._llm_thread = threading.Thread(target=run, name="LLM", daemon=True)
        self._llm_thread.start()

    def _check_llm(self, frame: np.ndarray) -> None:
        if not self._llm_answer:
            return
        move = self._llm_answer[0]
        if move is not None and move in self.board.legal_moves:
            self._accept(move, Tier.LLM, self.last_result, frame)
        else:
            self._ask_user(self._llm_candidates, frame)

    def _ask_user(self, candidates: list[chess.Move], frame: np.ndarray) -> None:
        if self.hooks.ask_user is not None:  # replay harness
            move = self.hooks.ask_user(self.board, candidates)
            if move is not None:
                self._accept(move, Tier.USER, self.last_result, frame)
            else:
                self._set_reference(frame)
                self.phase = Phase.PLAYING
            return
        self.choices = candidates
        self._ask_frame = frame
        self.phase = Phase.ASK_USER
        sans = ", ".join(f"{i + 1}) {self.board.san(m)}" for i, m in enumerate(candidates))
        self.message = f"Which move? {sans}  T type it  X no move  Z undo"
        logger.info("Tier 4: asking the player: %s", sans)

    def _promotion_piece(self, move: chess.Move, frame: np.ndarray) -> chess.Move:
        if move.promotion is None or self.hooks.choose_promotion is None:
            return move
        try:
            piece = self.hooks.choose_promotion(frame, self.cam, move.to_square, self.board.turn)
        except Exception:
            logger.exception("Promotion check failed; using a queen")
            piece = None
        if piece and piece != chess.QUEEN:
            logger.info("Promotion to %s", chess.piece_name(piece))
            return chess.Move(move.from_square, move.to_square, promotion=piece)
        return move

    def _accept(self, move: chess.Move, tier: Tier, result: MatchResult | None, frame: np.ndarray) -> None:
        move = self._promotion_piece(move, frame)
        self._ref_stack.append((self._ref, self._ref_image))
        san = self.record.add_move(move, int(tier), result.margin if result else 0.0)
        best = result.best if result and result.candidates else None
        log = MoveLog(len(self.board.move_stack), move.uci(), san, int(tier),
                      result.margin if result else 0.0, best.score if best else 0.0,
                      result.unexplained if result else 0.0, time.time())
        self.moves.append(log)
        if self._moves_csv:
            csv.writer(self._moves_csv).writerow([log.ply, log.uci, log.san, log.tier, f"{log.margin:.0f}",
                                                  f"{log.score:.0f}", f"{log.unexplained:.2f}", f"{log.timestamp:.3f}"])
            self._moves_csv.flush()
        logger.info("Move %d: %s (tier %d, margin %.0f, unexplained %.2f)",
                    log.ply, san, log.tier, log.margin, log.unexplained)
        self.last_tier = int(tier)
        self.choices = []
        self._set_reference(frame)
        self.phase = Phase.PLAYING
        self.message = f"{san}"
        for hook in self.hooks.on_move:
            hook(self.record)
        if self.hooks.analyse:
            self.hooks.analyse(self.board)
        self._moves_since_sync += 1
        if self._moves_since_sync >= self.s.health.sync_every_moves:
            self._moves_since_sync = 0
            self.check_sync(frame)
        if self.board.is_game_over():
            outcome = self.board.outcome()
            winner = outcome.winner if outcome else None
            self.end("draw" if winner is None else ("white" if winner else "black"))

    # ------------------------------------------------------------------ health / player actions

    def check_sync(self, frame: np.ndarray) -> list[str]:
        if self.cam is None:
            return []
        self.occupancy = self.occ_reader.read(self.cal.warp(frame), self.cam, self.board)
        report = compare_occupancy(self.occupancy, self.board, self.s.health.sync_min_confidence)
        self.desync = self.sync.update(report)
        if self.desync:
            self.message = f"Camera disagrees on {', '.join(self.desync)}. Z undo / T type a move / X ignore"
            logger.warning("Possible desync on %s (%s)", self.desync, report.mismatches)
        return self.desync

    def _answer_frame(self) -> np.ndarray:
        """The board as it was when the question was asked (the next move may already be on it)."""
        return self._ask_frame if self._ask_frame is not None else self._last_frame

    def _catch_up(self) -> None:
        """After the player answered, read any move made in the meantime."""
        frame = self._last_frame
        if frame is not None and self.gate is not None and self.gate.state == GateState.STABLE                 and frame is not self._ask_frame and self.phase == Phase.PLAYING:
            self._decide(self._read(frame), frame)

    def choose(self, index: int) -> None:
        if self.phase == Phase.ASK_USER and 0 <= index < len(self.choices):
            self._accept(self.choices[index], Tier.USER, self.last_result, self._answer_frame())
            self._ask_frame = None
            self._catch_up()

    def type_move(self, text: str) -> str | None:
        try:
            move = self.board.parse_san(text.strip())
        except ValueError:
            try:
                move = chess.Move.from_uci(text.strip().lower())
            except ValueError:
                return f"'{text}' isn't a move"
            if move not in self.board.legal_moves:
                return f"{text} isn't legal here"
        self._accept(move, Tier.USER, self.last_result, self._answer_frame())
        self._ask_frame = None
        self._catch_up()
        return None

    def ignore_change(self) -> None:
        """The board as it looks now is the current position (e.g. a piece was only adjusted)."""
        if self._last_frame is not None and self.reader is not None:
            self._set_reference(self._last_frame)
        self.choices, self._votes = [], []
        self.desync = []
        self.sync.reset()
        if self.phase not in (Phase.WAIT_START, Phase.ENDED):
            self.phase = Phase.PLAYING
        self.message = "OK, using the board as it is now."

    def undo(self) -> None:
        """Take back the last move. If the board still shows it, offer the alternatives."""
        if not self.record or not self.moves:
            return
        undone = self.record.undo()
        self.moves.pop()
        if self._ref_stack:
            self._ref, self._ref_image = self._ref_stack.pop()
        logger.info("Undid %s", undone.uci() if undone else None)
        self.phase = Phase.PLAYING
        self.message = f"Undid {undone.uci()}."
        if self._last_frame is not None:
            result = self._read(self._last_frame)
            if result.total_change >= self.s.reader.min_change_px and result.candidates:
                self._ask_user([c.move for c in result.top(3)], self._last_frame)

    def replace_last_move(self, move: chess.Move) -> None:
        """Swap the last recorded move for the right one (replay harness after a wrong accept)."""
        if not self.moves:
            return
        self.record.undo()
        log = self.moves[-1]
        log.san = self.record.add_move(move, log.tier, log.margin)
        log.uci = move.uci()

    def end(self, result: str) -> None:
        if not self.record or self.phase == Phase.ENDED:
            return
        self.record.finish(result)
        self.phase = Phase.ENDED
        self.message = f"Game over: {result}. PGN saved to {self.record.path.name}"
        logger.info("Game over (%s), %d plies, %s", result, len(self.moves), self.record.path)
        for hook in self.hooks.on_game_end:
            hook(self.record)

    def close(self) -> None:
        for f in (self._frames_csv, self._moves_csv):
            if f:
                f.close()
