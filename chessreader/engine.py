"""Stockfish analysis in a background thread so it never blocks the vision loop.

Only the newest position is analysed: if moves arrive faster than the engine finishes,
older positions are skipped. Results go to every registered listener.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import chess
import chess.engine

logger = logging.getLogger(__name__)


@dataclass
class Analysis:
    fen: str
    eval_cp: int | None            # centipawns from White's point of view
    mate: int | None               # moves to mate from White's point of view
    best_move: str | None          # SAN
    pv: list[str] = field(default_factory=list)  # SAN
    wdl: tuple[int, int, int] | None = None      # White win/draw/loss per mille
    depth: int | None = None
    lines: list[tuple[str, str]] = field(default_factory=list)  # (eval text, SAN line) for each multipv

    @property
    def eval_text(self) -> str:
        if self.mate is not None:
            return f"M{self.mate}" if self.mate > 0 else f"-M{-self.mate}"
        if self.eval_cp is None:
            return "?"
        return f"{self.eval_cp / 100:+.2f}"


def _score_text(score: chess.engine.PovScore) -> str:
    white = score.white()
    if white.is_mate():
        m = white.mate()
        return f"M{m}" if m > 0 else f"-M{-m}"
    return f"{white.score() / 100:+.2f}"


def _san_line(board: chess.Board, moves: list[chess.Move]) -> list[str]:
    b = board.copy(stack=False)
    out = []
    for move in moves:
        if not b.is_legal(move):
            break
        out.append(b.san(move))
        b.push(move)
    return out


class EngineWorker:
    def __init__(self, path: str | Path, think_time_s: float = 0.5, multipv: int = 3, threads: int = 2) -> None:
        self.path = str(path)
        self.limit = chess.engine.Limit(time=think_time_s)
        self.multipv = multipv
        self.threads = threads
        self.listeners: list[Callable[[Analysis], None]] = []
        self.latest: Analysis | None = None
        self._pending: chess.Board | None = None
        self._cond = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> "EngineWorker":
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Stockfish not found at {self.path}")
        self._thread = threading.Thread(target=self._run, name="Stockfish", daemon=True)
        self._thread.start()
        return self

    def submit(self, board: chess.Board) -> None:
        with self._cond:
            self._pending = board.copy()
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.path)
        except Exception as exc:
            logger.error("Couldn't start Stockfish (%s): %s", self.path, exc)
            return
        try:
            options = {"Threads": self.threads}
            if "UCI_ShowWDL" in engine.options:
                options["UCI_ShowWDL"] = True
            engine.configure(options)
            logger.info("Stockfish ready: %s", engine.id.get("name", self.path))
            while True:
                with self._cond:
                    while self._pending is None and not self._stop:
                        self._cond.wait()
                    if self._stop:
                        return
                    board, self._pending = self._pending, None
                if board.is_game_over():
                    continue
                try:
                    infos = engine.analyse(board, self.limit, multipv=self.multipv)
                except chess.engine.EngineError as exc:
                    logger.warning("Stockfish error: %s", exc)
                    continue
                result = self._to_analysis(board, infos)
                self.latest = result
                for listener in self.listeners:
                    try:
                        listener(result)
                    except Exception:
                        logger.exception("Analysis listener failed")
        finally:
            engine.quit()

    @staticmethod
    def _to_analysis(board: chess.Board, infos: list[chess.engine.InfoDict]) -> Analysis:
        top = infos[0]
        score = top.get("score")
        white = score.white() if score else None
        pv = _san_line(board, top.get("pv", []))
        wdl = None
        if top.get("wdl") is not None:
            w = top["wdl"].white()
            wdl = (w.wins, w.draws, w.losses)
        lines = [(_score_text(i["score"]), " ".join(_san_line(board, i.get("pv", [])[:6])))
                 for i in infos if i.get("score") is not None]
        return Analysis(
            fen=board.fen(),
            eval_cp=None if white is None or white.is_mate() else white.score(),
            mate=white.mate() if white is not None and white.is_mate() else None,
            best_move=pv[0] if pv else None,
            pv=pv,
            wdl=wdl,
            depth=top.get("depth"),
            lines=lines,
        )
