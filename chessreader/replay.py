"""Replay a recorded session (or video) through the same pipeline and score it.

    .\\.venv\\Scripts\\python run.py replay sessions\\20261017_013000 --pgn tests\\data\\test_game.pgn

Without a PGN it just reports what it read. With one, tier-4 questions are answered from
the PGN, wrong accepts are counted and then corrected so the rest of the game still counts.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import cv2
import numpy as np

from .calibration import Calibration, detect_board
from .confirm import LlmJudge, Tier
from .session import GameSession, Phase, SessionHooks
from .settings import Settings
from .storage import SessionRecorder

logger = logging.getLogger(__name__)


def frames_from(path: Path) -> Iterator[tuple[np.ndarray, float, str]]:
    """(image, timestamp, phase-at-recording) from a session folder or a video file."""
    if path.is_dir():
        with open(path / SessionRecorder.FRAMES_CSV, newline="", encoding="utf-8") as f:
            for name, ts, phase in csv.reader(f):
                image = cv2.imread(str(path / "frames" / name))
                if image is not None:
                    yield image, float(ts), phase
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Can't open video {path}")
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            yield image, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, "PLAYING"
    finally:
        cap.release()


@dataclass
class PlyReport:
    ply: int
    played: str
    truth: str | None
    tier: int
    margin: float

    @property
    def correct(self) -> bool | None:
        return None if self.truth is None else self.played == self.truth


def load_truth(pgn_path: Path | None) -> list[chess.Move]:
    if pgn_path is None:
        return []
    with open(pgn_path, encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
    return list(game.mainline_moves()) if game else []


def _game_start_index(frames: list[tuple[np.ndarray, float, str]]) -> int:
    """Where the live game started: the last frame recorded before play began."""
    for i, (_, _, phase) in enumerate(frames):
        if phase != Phase.WAIT_START.name:
            return max(0, i - 1)
    return 0


def _calibration_for(path: Path, start_image: np.ndarray, settings: Settings) -> Calibration:
    """The session's own calibration, or (for a video) the board found in its first frame."""
    saved = Calibration.load_saved(path) if path.is_dir() else None
    if saved is not None:
        return saved
    cal = detect_board(start_image, settings.calibration.warp_size)
    if cal is None:
        raise ValueError("Couldn't find the board in the first frame")
    return cal


def _record_new_moves(
    session: GameSession, before: int, truth: list[chess.Move], reports: list[PlyReport], rows: list[str]
) -> None:
    """Report the moves accepted since `before`; after a wrong accept, correct it so the rest still counts."""
    for log in session.moves[before:]:
        want = truth[log.ply - 1].uci() if log.ply - 1 < len(truth) else None
        report = PlyReport(log.ply, log.uci, want, log.tier, log.margin)
        reports.append(report)
        mark = "" if report.correct is None else ("ok" if report.correct else "WRONG")
        rows.append(
            f"{report.ply:>4} {report.played:>7} {report.truth or '-':>7} {report.tier:>4} {report.margin:>8.0f} {mark}"
        )
        if want is not None and log.uci != want:
            logger.warning(
                "ply %d: read %s, truth %s (tier %d, margin %.0f)", log.ply, log.uci, want, log.tier, log.margin
            )
            session.replace_last_move(chess.Move.from_uci(want))


def _summarize(reports: list[PlyReport], truth: list[chess.Move], frame_count: int) -> dict:
    auto = [r for r in reports if r.tier in (Tier.AUTO, Tier.VOTE, Tier.LLM)]
    return {
        "frames": frame_count,
        "moves_read": len(reports),
        "moves_in_truth": len(truth),
        "missed": max(0, len(truth) - len(reports)) if truth else None,
        "by_tier": {int(t): sum(1 for r in reports if r.tier == t) for t in Tier},
        "auto_accept_rate": (len(auto) / len(reports)) if reports else 0.0,
        "wrong_accepts": sum(1 for r in auto if r.correct is False),
    }


def _print_report(rows: list[str], final: list[chess.Move], summary: dict) -> None:
    print(f"\n{'ply':>4} {'read':>7} {'truth':>7} {'tier':>4} {'margin':>8}")
    for row in rows:
        print(row)
    moves = " ".join(m.uci() for m in final)
    print(f"\nfinal position after: {moves or '(no moves)'}")
    print(
        f"\nauto-accept rate {summary['auto_accept_rate']:.0%} | wrong accepts {summary['wrong_accepts']} | "
        f"tiers {summary['by_tier']} | read {summary['moves_read']}/{summary['moves_in_truth'] or '?'}"
    )


def run_replay(settings: Settings, path: Path, pgn_path: Path | None = None, use_llm: bool = False) -> dict:
    """Run a recorded session (or video) through the live pipeline; score it against a PGN if given."""
    truth = load_truth(pgn_path)
    # keep replay PGNs out of exports/, next to the recording instead
    out_dir = (path if path.is_dir() else path.parent) / "replay"
    settings = settings.model_copy(deep=True)
    settings.storage.exports_dir = str(out_dir.resolve())
    frames = list(frames_from(path))
    if not frames:
        raise ValueError(f"No frames in {path}")
    start = _game_start_index(frames)
    start_image = frames[start][0]
    cal = _calibration_for(path, start_image, settings)

    def ask(board: chess.Board, candidates: list[chess.Move]) -> chess.Move | None:
        ply = len(board.move_stack)
        return truth[ply] if ply < len(truth) else None

    llm = LlmJudge(settings.llm.model, True) if use_llm else None
    session = GameSession(settings, cal, SessionHooks(ask_user=ask), llm)
    error = session.start_game(start_image)
    if error:
        raise ValueError(error)

    reports: list[PlyReport] = []
    rows: list[str] = []  # the printed table, in order, takebacks included
    for image, ts, _ in frames[start + 1 :]:
        before = len(session.moves)
        last = session.moves[-1].uci if session.moves else None
        session.process(image, ts)
        if len(session.moves) < before:  # put back by hand (or undone)
            rows.append(f"{before:>4} {last:>7} {'':>7} {'':>4} {'':>8} taken back")
            before = len(session.moves)
        while session.phase == Phase.WAIT_LLM:  # replay waits for the LLM instead of streaming on
            session.wait_for_llm()
            session.process(image, ts)
        _record_new_moves(session, before, truth, reports, rows)
        if session.phase == Phase.ENDED:
            break

    summary = _summarize(reports, truth, len(frames) - start)
    _print_report(rows, session.board.move_stack, summary)
    return summary
