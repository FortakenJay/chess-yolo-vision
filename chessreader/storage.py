"""Game storage: a PGN file in exports/, rewritten after every move."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.pgn

logger = logging.getLogger(__name__)

PGN_RESULTS = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2", "unknown": "*"}


class GameRecord:
    def __init__(self, exports_dir: Path, white: str = "White", black: str = "Black",
                 event: str = "Casual Game", site: str = "ESP32 chess reader",
                 starting_fen: str = chess.STARTING_FEN) -> None:
        self.started_at = datetime.now(timezone.utc)
        self.starting_fen = starting_fen
        self.board = chess.Board(starting_fen)
        self.game = chess.pgn.Game()
        h = self.game.headers
        h["Event"], h["Site"], h["White"], h["Black"] = event, site, white, black
        h["Date"] = self.started_at.astimezone().strftime("%Y.%m.%d")
        h["Time"] = self.started_at.astimezone().strftime("%H:%M:%S")
        h["Result"] = "*"
        if starting_fen != chess.STARTING_FEN:
            self.game.setup(chess.Board(starting_fen))
        self.node: chess.pgn.GameNode = self.game
        self.result = "unknown"
        exports_dir.mkdir(parents=True, exist_ok=True)
        self.path = exports_dir / f"game_{self.started_at.astimezone():%Y%m%d_%H%M%S}.pgn"
        self.write()

    @property
    def total_moves(self) -> int:
        return len(self.board.move_stack)

    def add_move(self, move: chess.Move, tier: int, margin: float) -> str:
        san = self.board.san(move)
        self.board.push(move)
        self.node = self.node.add_variation(move)
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        self.node.comment = f"[%ts {stamp}] tier {tier}"
        self.write()
        return san

    def undo(self) -> chess.Move | None:
        if not self.board.move_stack:
            return None
        move = self.board.pop()
        parent = self.node.parent
        if parent is not None:
            parent.remove_variation(self.node)
            self.node = parent
        self.write()
        return move

    def finish(self, result: str) -> None:
        self.result = result if result in PGN_RESULTS else "unknown"
        self.game.headers["Result"] = PGN_RESULTS[self.result]
        self.write()

    def pgn(self) -> str:
        return str(self.game)

    def write(self) -> None:
        """Save the PGN. A failed save is logged and retried on the next move; it never stops the game."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(self.pgn() + "\n", encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.error("Couldn't save %s: %s", self.path, exc)

