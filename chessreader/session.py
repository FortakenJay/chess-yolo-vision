"""One game: frames in, accepted moves out. No UI here, so live play and replay share it.

Per frame: the motion gate decides when the board is still. On each new still board the
change since the last accepted position is matched against every legal move, then the
confirmation tiers decide (auto, vote over re-reads, LLM, or ask the player).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import chess
import cv2
import numpy as np

from .calibration import Calibration, detect_board, edge_map
from .confirm import ConfirmSettings, LlmJudge, Tier, average_results, ends_ok, tier1_ok, tier2_ok
from .engine import Analysis
from .gating import GateState, GateUpdate, MotionGate
from .geometry import BoardCamera, orient_from_start
from .health import DriftMonitor, SyncMonitor, compare_occupancy
from .matcher import MatchResult, MoveMatcher
from .openings import book_for
from .reader import ChangeReader, OccupancyReader, Prepared
from .settings import Settings
from .storage import GameRecord, MoveLog, SessionRecorder
from .tracking import CameraTracker

logger = logging.getLogger(__name__)


class Phase(StrEnum):
    WAIT_START = "set up the start position, press S"
    PLAYING = "playing"
    VOTING = "re-reading"
    WAIT_LLM = "asking the LLM"
    ASK_USER = "which move? press 1-3"
    PAUSED = "paused"
    ENDED = "game over"


@dataclass
class SessionHooks:
    """Outside services: callbacks a front end plugs in (all optional)."""

    on_move: list[Callable[[GameRecord], None]] = field(default_factory=list)
    on_game_start: list[Callable[[GameRecord], None]] = field(default_factory=list)
    on_game_end: list[Callable[[GameRecord], None]] = field(default_factory=list)
    analyse: Callable[[chess.Board], None] | None = None  # asynchronous: returns at once
    # (frame, camera, square, colour) -> piece type, or None to keep the default queen
    choose_promotion: Callable[[np.ndarray, BoardCamera, int, bool], int | None] | None = None
    ask_user: Callable[[chess.Board, list[chess.Move]], chess.Move | None] | None = None  # replay only


@dataclass
class Pipeline:
    """Everything built from one calibration: rebuilt together whenever the board or camera moves."""

    cam: BoardCamera
    matcher: MoveMatcher
    reader: ChangeReader
    gate: MotionGate
    region_px: float  # area moves can change, in full-resolution pixels

    @classmethod
    def build(cls, cal: Calibration, settings: Settings, tone_q: tuple[float, float] | None) -> Pipeline:
        r = settings.reader
        cam = BoardCamera.from_calibration(cal, r.height_scale)
        roi = cam.roi()
        matcher = MoveMatcher(cam, roi, r.scale, r.q_in, r.q_out, r.q_in_low, r.off_board_weight)
        if tone_q is not None:
            matcher.set_tone_contrast(*tone_q)
        reader = ChangeReader(roi, r.scale, t_low=r.t_low, t_high=r.t_high, texture_weight=r.texture_weight)
        x0, y0, x1, y1 = roi
        region_full = cv2.resize(matcher.region.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        g = settings.gate
        gate = MotionGate(region_full, g.pixel_threshold, g.motion_fraction, g.stable_time_s, g.min_stable_frames)
        return cls(cam, matcher, reader, gate, float(matcher.region.sum()) * matcher.px)


class GameSession:
    def __init__(
        self,
        settings: Settings,
        cal: Calibration,
        hooks: SessionHooks | None = None,
        llm: LlmJudge | None = None,
        session_dir: Path | None = None,
    ) -> None:
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

        self._pipe: Pipeline | None = None  # built when a game starts
        self.occ_reader = OccupancyReader()
        self.sync = SyncMonitor()
        self.drift: DriftMonitor | None = None
        self.tracker = CameraTracker()
        self._tone_q: tuple[float, float] | None = None  # change rates for black-on-dark, white-on-light

        # the frame of the last accepted position, and its prepared form (set by start_game)
        self._ref: Prepared
        self._ref_image: np.ndarray
        self._ref_stack: list[tuple[Prepared, np.ndarray]] = []
        self._votes: list[MatchResult] = []
        self._last_frame: np.ndarray | None = None
        self._last_drift_t = 0.0
        self._moves_since_sync = 0
        self._llm_thread: threading.Thread | None = None
        self._llm_answer: list = []
        self._ask_frame: np.ndarray | None = None
        self._vote_frame: np.ndarray | None = None
        self._ref_provisional = False  # reference taken when a question was asked; a hand may be in it

        self.recorder = (
            SessionRecorder(session_dir, settings.storage.record_frames) if session_dir is not None else None
        )

    # ------------------------------------------------------------------ properties

    @property
    def board(self) -> chess.Board:
        return self.record.board if self.record else chess.Board()

    @property
    def ready(self) -> bool:
        """A game has started, so the camera model, matcher, reader and gate exist."""
        return self._pipe is not None

    @property
    def pipe(self) -> Pipeline:
        if self._pipe is None:
            raise RuntimeError("No game started yet: the board pipeline isn't built")
        return self._pipe

    @property
    def cam(self) -> BoardCamera:
        return self.pipe.cam

    @property
    def matcher(self) -> MoveMatcher:
        return self.pipe.matcher

    @property
    def reader(self) -> ChangeReader:
        return self.pipe.reader

    @property
    def gate(self) -> MotionGate:
        return self.pipe.gate

    @property
    def gate_state(self) -> GateState | None:
        return self.gate.state if self.ready else None

    # ------------------------------------------------------------------ setup

    def _build(self, cal: Calibration) -> None:
        self.cal = cal
        self._pipe = Pipeline.build(cal, self.s, self._tone_q)

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.matcher.roi
        return frame[y0:y1, x0:x1]

    def start_game(self, frame: np.ndarray, board: chess.Board | None = None) -> str | None:
        """Orient the board from the position on it and start recording. Returns an error or None."""
        position = board or chess.Board()
        bases = [self.cal]
        if board is None:
            fresh = self._fresh_calibration(frame)
            if fresh is not self.cal:
                bases.insert(0, fresh)
        oriented, error = [], ""
        for base in bases:
            try:
                oriented.append(orient_from_start(base, frame, self.s.reader.height_scale, position))
            except ValueError as exc:
                error = str(exc)
        if not oriented:
            return error
        # the board found now usually wins; if it snapped a rank off (pieces hide corners), the saved one fits better
        cal, _, contrast = max(oriented, key=lambda o: o[2])
        if len(oriented) > 1:
            logger.info(
                "Orientation contrast: board found now %.0f, saved calibration %.0f", oriented[0][2], oriented[1][2]
            )
        if contrast < 25:
            return f"Can't see the pieces clearly (contrast {contrast:.0f}). Is the start position set up?"
        self._build(cal)
        if self.recorder is not None:
            self.recorder.save_calibration(cal)
        self.drift = DriftMonitor(
            edge_map(cal.warp(frame), self.s.calibration.canny_low, self.s.calibration.canny_high),
            self.s.calibration.canny_low,
            self.s.calibration.canny_high,
            threshold=self.s.health.drift_threshold,
        )
        warped = cal.warp(frame)
        self.occ_reader.fit(warped, self.cam, position)
        self._learn_tone_contrast()
        self._update_occupancy(frame)
        st = self.s.storage
        from .settings import resolve_path

        self.record = GameRecord(
            resolve_path(st.exports_dir),
            st.white,
            st.black,
            starting_fen=position.fen(),
            opening_book=book_for(self.s.openings),
        )
        self._set_reference(frame)
        self._ref_stack.clear()
        self.moves.clear()
        self.phase = Phase.PLAYING
        self.message = f"Game started (white at {self._white_side()}). Make the first move."
        logger.info("Game started; orientation contrast %.0f; PGN %s", contrast, self.record.path)
        for hook in self.hooks.on_game_start:
            hook(self._game())
        if self.hooks.analyse:
            self.hooks.analyse(self.board)
        return None

    def _learn_tone_contrast(self) -> None:
        """How visible a piece is on a square of its own tone, on this board in this light."""
        r = self.s.reader
        qs = []
        for piece, square in (("black", "dark"), ("white", "light")):
            d = self.occ_reader.contrast(piece, square)
            qs.append(self.matcher.q_in if d is None else self.matcher.q_for_contrast(d, r.t_low, r.t_high))
            logger.info(
                "%s pieces on %s squares: contrast %s, q_in %.2f",
                piece,
                square,
                "unknown" if d is None else f"{d:.0f}",
                qs[-1],
            )
        self._tone_q = (qs[0], qs[1])
        self.matcher.set_tone_contrast(*self._tone_q)

    def _fresh_calibration(self, frame: np.ndarray) -> Calibration:
        """Find the board again in this frame; the saved calibration is stale if the camera moved."""
        found = detect_board(frame, self.s.calibration.warp_size)
        if found is None or not found.fits(self.s.calibration.max_reprojection_px):
            logger.info("Board not re-detected at start; using the saved calibration")
            return self.cal
        old, new = np.asarray(self.cal.outer_corners), np.asarray(found.outer_corners)
        shift = float(np.linalg.norm(old[:, None] - new[None], axis=2).min(axis=1).mean())
        if shift > 2.0:
            logger.info("Board is %.0f px from the saved calibration", shift)
        return found

    def _follow_camera(self, frame: np.ndarray) -> bool:
        """If the camera shifted since the reference frame, move the calibration and reference with it."""
        shift = self.tracker.estimate(self._ref_image, frame, np.asarray(self.cal.outer_corners))
        if shift is None or shift.moved_px < self.s.health.track_min_px:
            return False
        if shift.moved_px > 3 * self.cal.square_px:
            logger.info("Camera tracking: shift of %.0f px is too large to trust", shift.moved_px)
            return False
        h, w = frame.shape[:2]
        ref_image = cv2.warpPerspective(self._ref_image, shift.matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        cal = self.cal.moved(shift.matrix)
        self._build(cal)
        self._set_reference(ref_image)
        if self.drift is not None:
            self.drift.reset(edge_map(cal.warp(ref_image), self.s.calibration.canny_low, self.s.calibration.canny_high))
        logger.info("Camera moved %.1f px; followed it (%d feature matches)", shift.moved_px, shift.inliers)
        return True

    def _took_back(self, result: MatchResult, frame: np.ndarray) -> bool:
        """The player put the last move back by hand.

        Taking a move back changes exactly the pixels the move itself changed, so the change since
        the last accepted position is scored against the last move's own mask. It has to explain
        the change clearly better than any legal next move.
        """
        if not self.moves:
            return False
        before = self.board.copy()
        last = before.pop()
        cand, unexplained = self.matcher.score_move(before, last, result.change)
        best = result.best.score if result.best else 0.0
        logger.debug(
            "takeback check: undoing %s scores %.0f (unexplained %.2f), best next move %.0f",
            last.uci(),
            cand.score,
            unexplained,
            best,
        )
        if (
            cand.score < self.confirm.min_score
            or cand.score - best < self.confirm.t2_margin
            or unexplained > self.confirm.max_unexplained
            or not ends_ok(cand, self.confirm)
        ):
            return False
        undone = self.moves[-1].san
        self.undo(reread=False)
        self._set_reference(frame)
        self._update_occupancy(frame)
        self.choices = []
        self.phase = Phase.PLAYING
        self.message = f"{undone} was taken back."
        logger.info("Move %d (%s) was taken back on the board", len(self.moves) + 1, undone)
        return True

    def _adjusted(self, result: MatchResult, frame: np.ndarray) -> bool:
        """A piece was straightened or nudged in place (e.g. one knocked over): no move, no question."""
        adj = self.matcher.best_adjustment(self.board, result.change)
        if adj is None:
            return False
        square, score, unexplained = adj
        best = result.best.score if result.best else 0.0
        # half the tier-2 margin: the destination check below already rules the move out
        if (
            score < self.confirm.min_score
            or score - best < self.confirm.t2_margin / 2
            or unexplained > self.confirm.max_unexplained
            or result.off_board > self.confirm.max_off_board
        ):
            return False
        # only when the piece's likeliest move would have had to show up clearly where it went and
        # didn't: a black piece arriving on a dark square can look like nothing arrived
        moves = [c for c in result.candidates if c.move.from_square == square]
        if moves and not (
            moves[0].arrived_q >= self.confirm.clear_end_q and moves[0].arrived_ratio < self.confirm.min_end_ratio
        ):
            return False
        logger.info(
            "Piece on %s adjusted in place (%.0f vs best move %.0f); no move", chess.square_name(square), score, best
        )
        self._set_reference(frame)
        self._update_occupancy(frame)
        self.message = f"Piece on {chess.square_name(square)} adjusted; no move."
        return True

    def _white_side(self) -> str:
        """Where white sits in the camera image, in words (for the LLM and the HUD)."""
        x, y = self.cam.project(np.array([[4.0, 8.0, 0.0]]))[0]  # middle of white's back edge
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
        if self.drift is not None:
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
        logger.debug(
            "read: change %.0fpx, best %s %.0f, margin %.0f, unexplained %.2f, off board %.0fpx",
            result.total_change,
            best.move.uci() if best else None,
            best.score if best else 0,
            result.margin,
            result.unexplained,
            result.off_board,
        )
        return result

    def process(self, frame: np.ndarray, timestamp: float) -> None:
        """Take one camera frame: record it, update the motion gate, and act on it for the current phase."""
        if self.recorder is not None:
            self.recorder.frame(frame, timestamp, self.phase.name)
        self._last_frame = frame
        if self.phase in (Phase.WAIT_START, Phase.ENDED) or not self.ready:
            return
        update = self.gate.update(self._crop(frame), timestamp)  # the gate keeps watching even while paused
        if self.phase == Phase.PLAYING:
            self._on_playing(frame, timestamp, update)
        elif self.phase == Phase.VOTING:
            self._on_voting(frame, update)
        elif self.phase == Phase.ASK_USER:
            self._on_question(frame, update)
        elif self.phase == Phase.WAIT_LLM:
            self._check_llm(frame)
        # PAUSED: wait for C

    def _on_playing(self, frame: np.ndarray, timestamp: float, update: GateUpdate) -> None:
        if update.became_stable and self._follow_camera(frame):
            self.gate.update(self._crop(frame), timestamp)  # the new gate starts from this frame
            self._last_drift_t = timestamp
        elif update.state == GateState.STABLE and timestamp - self._last_drift_t > self.s.health.drift_every_s:
            self._last_drift_t = timestamp
            if self.drift is not None and self.drift.update(self.cal.warp(frame)) and not self._follow_camera(frame):
                self.phase = Phase.PAUSED
                self.message = "The board or camera moved. Press C to recalibrate (pieces can stay)."
                return
        if update.became_stable:
            self._decide(self._read(frame), frame)

    def _on_voting(self, frame: np.ndarray, update: GateUpdate) -> None:
        if update.state == GateState.MOVING:
            # the next move is starting: settle this one with the reads we have
            self._decide_after_vote(self._vote_frame if self._vote_frame is not None else frame)
        elif update.state == GateState.STABLE:
            self._votes.append(self._read(frame))
            self._vote_frame = frame
            if len(self._votes) > self.confirm.vote_frames:
                self._decide_after_vote(frame)

    def _on_question(self, frame: np.ndarray, update: GateUpdate) -> None:
        """A question is open; the player may also settle it on the board instead of answering."""
        if not update.became_stable:
            return
        result = self._read(frame)
        if result.total_change < self.s.reader.min_change_px:
            self.choices = []
            self.phase = Phase.PLAYING
            self.message = "Board matches the last position again."
        elif result.best is not None and tier1_ok(result, self.confirm):  # the hand that caused it is gone
            self._ask_frame = None
            self._accept(result.best.move, Tier.AUTO, result, frame)
        elif self._took_back(result, frame):
            self.choices = []
        elif self._adjusted(result, frame):
            self.choices = []
            self._ask_frame = None
            self.phase = Phase.PLAYING

    def _decide(self, result: MatchResult, frame: np.ndarray) -> None:
        """What a newly still board means, in order: nothing, covered, takeback, adjustment, a move."""
        r = self.s.reader
        if self._question_frame_unusable(result, frame) or result.total_change < r.min_change_px:
            return  # same position as before (or the reference was just replaced)
        if result.total_change > r.occlusion_fraction * self.pipe.region_px:
            self.message = "Something is covering the board."
            return
        if self._took_back(result, frame) or self._adjusted(result, frame):
            return
        best = result.best
        if best is None or best.score < self.confirm.min_score:
            self._no_confident_move(result, frame)
        elif tier1_ok(result, self.confirm):
            self._accept(best.move, Tier.AUTO, result, frame)
        else:
            self._start_vote(result, frame, f"Not sure ({best.move.uci()} by {result.margin:.0f}); re-reading...")

    def _question_frame_unusable(self, result: MatchResult, frame: np.ndarray) -> bool:
        """First still board after an answer: a clean next move counts; anything else means the frame from
        question time was unusable (a hand on the board), so the board as it is now becomes the reference."""
        if not self._ref_provisional:
            return False
        self._ref_provisional = False
        if result.total_change < self.s.reader.min_change_px or tier1_ok(result, self.confirm):
            return False
        logger.debug(
            "reference from question time doesn't fit (%.0fpx); using the board as it is now", result.total_change
        )
        self._set_reference(frame)
        return True

    def _no_confident_move(self, result: MatchResult, frame: np.ndarray) -> None:
        """No move scores min_score: re-read a faint but clear move, wait for a sleeve, ignore drift, or ask."""
        best = result.best
        if best is not None and best.score > 0 and result.margin >= self.confirm.t2_margin:
            # weak but clearly the best explanation (black on dark squares): re-read, never drop it
            self._start_vote(result, frame, f"Faint change ({best.move.uci()}); re-reading...")
        elif result.off_board >= self.confirm.wait_off_board:
            # no move fits and the change runs off the board's edge: a sleeve or hand resting there
            self.message = "Something is resting at the board's edge; waiting."
            logger.debug(
                "waiting: %.0fpx changed, %.0fpx of it beyond the board edge", result.total_change, result.off_board
            )
        elif result.total_change < self.s.reader.unexplained_alarm_px:
            # e.g. lighting / exposure drift: no move explains it, so nothing happened
            logger.debug("ignoring %.0fpx of change that no move explains", result.total_change)
        else:
            logger.info("Board changed (%.0fpx) but no legal move fits", result.total_change)
            self._ask_user([c.move for c in result.top(3)], frame)
            self.message = "The board changed but no legal move fits. " + self.message

    def _start_vote(self, result: MatchResult, frame: np.ndarray, message: str) -> None:
        """Tier 2: re-read a few more still frames before deciding."""
        self._votes = [result]
        self._vote_frame = frame
        self.phase = Phase.VOTING
        self.message = message

    def _decide_after_vote(self, frame: np.ndarray) -> None:
        merged = average_results(self._votes)
        self._votes = []
        self.last_result = merged
        if merged.best is not None and tier2_ok(merged, self.confirm):
            self._accept(merged.best.move, Tier.VOTE, merged, frame)
            return
        candidates = [c.move for c in merged.top(3)]
        if self.llm is not None and self.llm.available:
            self._start_llm(self.llm, candidates, frame)
            return
        self._ask_user(candidates, frame)

    def _start_llm(self, llm: LlmJudge, candidates: list[chess.Move], frame: np.ndarray) -> None:
        self.phase = Phase.WAIT_LLM
        self.message = "Asking the LLM..."
        self._llm_answer = []
        board, before, after = self.board.copy(), self._ref_image.copy(), frame.copy()
        note = f"White sits at {self._white_side()}."

        def run() -> None:
            self._llm_answer.append(llm.choose(board, before, after, candidates, note))

        self._llm_candidates = candidates
        self._llm_thread = threading.Thread(target=run, name="LLM", daemon=True)
        self._llm_thread.start()

    def wait_for_llm(self) -> None:
        """Block until a pending tier-3 answer is in (replay waits instead of streaming on)."""
        if self._llm_thread is not None:
            self._llm_thread.join()

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
                return
            # no answer known: leave the question open, like a player who doesn't press a key, so later
            # reads can still settle it (a clean re-read, a takeback, an adjusted piece)
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

    def _game(self) -> GameRecord:
        """The game being recorded; only called while one is (a programming error otherwise)."""
        if self.record is None:
            raise RuntimeError("No game in progress")
        return self.record

    def _accept(self, move: chess.Move, tier: Tier, result: MatchResult | None, frame: np.ndarray) -> None:
        move = self._promotion_piece(move, frame)
        self._ref_provisional = False
        self._ref_stack.append((self._ref, self._ref_image))
        san = self._game().add_move(move, int(tier), result.margin if result else 0.0)
        best = result.best if result and result.candidates else None
        log = MoveLog(
            len(self.board.move_stack),
            move.uci(),
            san,
            int(tier),
            result.margin if result else 0.0,
            best.score if best else 0.0,
            result.unexplained if result else 0.0,
            time.time(),
        )
        self.moves.append(log)
        if self.recorder is not None:
            self.recorder.move(log)
        logger.info(
            "Move %d: %s (tier %d, margin %.0f, unexplained %.2f)", log.ply, san, log.tier, log.margin, log.unexplained
        )
        self.last_tier = int(tier)
        self.choices = []
        self._set_reference(frame)
        self.phase = Phase.PLAYING
        self.message = f"{san}"
        for hook in self.hooks.on_move:
            hook(self._game())
        if self.hooks.analyse:
            self.hooks.analyse(self.board)
        self._moves_since_sync += 1
        if self._moves_since_sync >= self.s.health.sync_every_moves:
            self._moves_since_sync = 0
            self.check_sync(frame)
        else:
            self._update_occupancy(frame)
        if self.board.is_game_over():
            outcome = self.board.outcome()
            winner = outcome.winner if outcome else None
            self.end("draw" if winner is None else ("white" if winner else "black"))

    # ------------------------------------------------------------------ health / player actions

    def _update_occupancy(self, frame: np.ndarray) -> None:
        """What the camera sees on each square (the Board window's dots), refreshed after every move."""
        if self.ready and self.occ_reader.templates:
            self.occupancy = self.occ_reader.read(self.cal.warp(frame), self.cam, self.board)

    def check_sync(self, frame: np.ndarray) -> list[str]:
        if not self.ready:
            return []
        self._update_occupancy(frame)
        if self.occupancy is None:
            return []
        report = compare_occupancy(self.occupancy, self.board, self.s.health.sync_min_confidence)
        self.desync = self.sync.update(report)
        if self.desync:
            self.message = f"Camera disagrees on {', '.join(self.desync)}. Z undo / T type a move / X ignore"
            logger.warning("Possible desync on %s (%s)", self.desync, report.mismatches)
        return self.desync

    def _answer_frame(self) -> np.ndarray:
        """The board as it was when the question was asked (the next move may already be on it)."""
        frame = self._ask_frame if self._ask_frame is not None else self._last_frame
        if frame is None:  # an answer before any frame arrived can't happen: no question is open yet
            raise RuntimeError("No frame to apply the answer to")
        return frame

    def _catch_up(self) -> None:
        """After the player answered, read any move made in the meantime."""
        frame = self._last_frame
        if frame is None or self.phase != Phase.PLAYING or frame is self._ask_frame:
            return
        self._ref_provisional = True
        self._follow_camera(frame)  # the camera may have moved since the question was asked
        if self.ready and self.gate.state == GateState.STABLE:
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
        if self._last_frame is not None and self.ready:
            self._set_reference(self._last_frame)
        self.choices, self._votes = [], []
        self.desync = []
        self.sync.reset()
        if self.phase not in (Phase.WAIT_START, Phase.ENDED):
            self.phase = Phase.PLAYING
        self.message = "OK, using the board as it is now."

    def undo(self, reread: bool = True) -> None:
        """Take back the last move. If the board still shows it, offer the alternatives."""
        if not self.record or not self.moves:
            return
        undone = self.record.undo()
        self.moves.pop()
        if self._ref_stack:
            _, image = self._ref_stack.pop()
            self._set_reference(image)  # re-prepare: the calibration may have changed since
            if self._last_frame is not None:
                self._follow_camera(self._last_frame)
        name = undone.uci() if undone else "?"
        logger.info("Undid %s", name)
        self.phase = Phase.PLAYING
        self.message = f"Undid {name}."
        if reread and self._last_frame is not None:
            result = self._read(self._last_frame)
            if result.total_change >= self.s.reader.min_change_px and result.candidates:
                self._ask_user([c.move for c in result.top(3)], self._last_frame)

    def replace_last_move(self, move: chess.Move) -> None:
        """Swap the last recorded move for the right one (replay harness after a wrong accept)."""
        if not self.moves:
            return
        record = self._game()
        record.undo()
        log = self.moves[-1]
        log.san = record.add_move(move, log.tier, log.margin)
        log.uci = move.uci()

    def end(self, result: str) -> None:
        if not self.record or self.phase == Phase.ENDED:
            return
        self.record.finish(result)
        self.phase = Phase.ENDED
        self.message = f"Game over: {result}. PGN saved to {self.record.path.name}"
        logger.info("Game over (%s), %d plies, %s", result, len(self.moves), self.record.path)
        for hook in self.hooks.on_game_end:
            hook(self._game())
