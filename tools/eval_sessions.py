"""Score recorded sessions frame by frame against what was really on the board.

    .venv/bin/python tools/eval_sessions.py                 # every session in session_timelines.json
    .venv/bin/python tools/eval_sessions.py 20261017_104913 -v

Unlike `run.py replay --pgn`, this copes with moves taken back by hand: each session's truth is a
timeline of (first frame, last frame, moves from the start) in `session_timelines.json`. The game
is driven like a live player would: a question is answered only when the real move is offered,
three frames after it appears. On every frame where the board is still and no question is open,
the program's position must equal the truth. The target is 0 wrong frames.

Sessions live in `sessions/` (git-ignored), so this runs only where they were recorded or copied.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chessreader.calibration import Calibration  # noqa: E402
from chessreader.gating import GateState  # noqa: E402
from chessreader.replay import frames_from  # noqa: E402
from chessreader.session import GameSession, Phase  # noqa: E402
from chessreader.settings import load_settings, resolve_path  # noqa: E402

TIMELINES = Path(__file__).with_name("session_timelines.json")
ANSWER_AFTER_FRAMES = 3  # the player reads the question before answering


@dataclass
class Score:
    name: str
    still_frames: int = 0
    wrong_frames: list[int] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    final: list[str] = field(default_factory=list)

    def line(self) -> str:
        return (
            f"{self.name}: wrong-state still frames {len(self.wrong_frames)}/{self.still_frames}"
            f"  questions at {self.questions}  final {' '.join(self.final) or '(no moves)'}"
        )


def truth_positions(timeline: list[dict]) -> list[tuple[int, int, str]]:
    """(first frame, last frame, board FEN) per stretch, computed once."""
    out = []
    for stretch in timeline:
        board = chess.Board()
        for san in stretch["moves"]:
            board.push_san(san)
        out.append((stretch["first"], stretch["last"], board.board_fen()))
    return out


def truth_at(positions: list[tuple[int, int, str]], frame: int) -> str | None:
    """The real position (board FEN) at a frame, or None between known stretches (a hand moving)."""
    return next((fen for first, last, fen in positions if first <= frame <= last), None)


def evaluate(name: str, timeline: list[dict]) -> Score:
    settings = load_settings()
    settings.engine.enabled = False
    folder = resolve_path(settings.storage.sessions_dir) / name
    settings.storage.exports_dir = str(folder / "eval")
    session = GameSession(settings, Calibration.load(folder))
    frames = enumerate(frames_from(folder))  # streamed, not all decoded up front
    _, (first, _, _) = next(frames)
    error = session.start_game(first)
    if error:
        raise SystemExit(f"{name}: game didn't start: {error}")

    positions = truth_positions(timeline)
    score, asked_at = Score(name), None
    for i, (image, ts, _) in frames:
        session.process(image, ts)
        truth = truth_at(positions, i)
        if session.phase == Phase.ASK_USER:
            if asked_at is None:
                asked_at = i
                score.questions.append(str(i))
            if truth is not None and i - asked_at >= ANSWER_AFTER_FRAMES:
                for k, move in enumerate(session.choices):
                    after = session.board.copy()
                    after.push(move)
                    if after.board_fen() == truth:
                        session.choose(k)
                        score.questions[-1] += "->answered"
                        break
        if session.phase != Phase.ASK_USER:
            asked_at = None
        if truth is not None and session.gate_state == GateState.STABLE and session.phase == Phase.PLAYING:
            score.still_frames += 1
            if session.board.board_fen() != truth:
                score.wrong_frames.append(i)
    score.final = [m.uci() for m in session.board.move_stack]
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sessions", nargs="*", help="session folder names (default: all with a timeline)")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every read and decision")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(name)s %(message)s")
    logging.getLogger("chessreader.session").setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    timelines = {k: v["timeline"] for k, v in json.loads(TIMELINES.read_text(encoding="utf-8")).items()}
    names = args.sessions or list(timelines)
    failed = False
    for name in names:
        if name not in timelines:
            raise SystemExit(f"No timeline for {name} in {TIMELINES.name}")
        score = evaluate(name, timelines[name])
        print(score.line())
        if score.wrong_frames:
            failed = True
            print("  first wrong frames:", score.wrong_frames[:12])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
