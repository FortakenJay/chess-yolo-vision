"""Game storage: the PGN in exports/ (rewritten after every move) and the session recording for replay."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import chess
import chess.pgn
import cv2
import numpy as np

from .openings import load_book

if TYPE_CHECKING:
    from .calibration import Calibration
    from .openings import OpeningBook

logger = logging.getLogger(__name__)


DEFAULT_BOOK: Literal["default"] = "default"  # GameRecord(opening_book=...): the bundled book; None = no tags

PGN_RESULTS = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2", "unknown": "*"}


class GameRecord:
    def __init__(
        self,
        exports_dir: Path,
        white: str = "White",
        black: str = "Black",
        event: str = "Casual Game",
        site: str = "ESP32 chess reader",
        starting_fen: str = chess.STARTING_FEN,
        opening_book: OpeningBook | Literal["default"] | None = DEFAULT_BOOK,
    ) -> None:
        self.started_at = datetime.now(UTC)
        self.opening_book = opening_book
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
        self.path = self._free_path(exports_dir, f"game_{self.started_at.astimezone():%Y%m%d_%H%M%S}")
        self.write()

    @staticmethod
    def _free_path(directory: Path, stem: str) -> Path:
        """A PGN path not used yet: a reset (R) within the same second must not overwrite the last game."""
        path, n = directory / f"{stem}.pgn", 2
        while path.exists():
            path, n = directory / f"{stem}_{n}.pgn", n + 1
        return path

    def add_move(self, move: chess.Move, tier: int, margin: float) -> str:
        san = self.board.san(move)
        self.board.push(move)
        self.node = self.node.add_variation(move)
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        self.node.comment = f"[%ts {stamp}] tier {tier}"
        self._tag_opening()
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
        self._tag_opening()
        self.write()
        return move

    def _tag_opening(self) -> None:
        """Set the ECO and Opening headers to the deepest named book position the game reached."""
        h = self.game.headers
        name = eco = None
        try:
            book = load_book() if self.opening_book == DEFAULT_BOOK else self.opening_book
            if book is not None:
                info = book.lookup(self.board)
                eco, name = info.eco, info.name
        except Exception as exc:
            logger.warning("Opening lookup failed: %s", exc)
        if eco and name:
            h["ECO"], h["Opening"] = eco, name
        else:
            for key in ("ECO", "Opening"):
                if key in h:
                    del h[key]

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


@dataclass
class MoveLog:
    """One accepted move, as listed in moves.csv."""

    ply: int
    uci: str
    san: str
    tier: int
    margin: float
    score: float
    unexplained: float
    timestamp: float


class SessionRecorder:
    """One game's frames and accepted moves in a session folder, for `replay`.

    Each row is appended and the file closed straight away, so a crash loses at most the frame
    being written, and nothing stays open between frames.
    """

    FRAMES_CSV = "frames.csv"  # image name, timestamp, phase (read back by replay.frames_from)
    MOVES_CSV = "moves.csv"
    MOVE_COLUMNS = ("ply", "uci", "san", "tier", "margin", "score", "unexplained", "t")

    def __init__(self, directory: Path, record_frames: bool = True) -> None:
        self.directory = directory
        self.record_frames = record_frames
        self._frame_no = 0
        (directory / "frames").mkdir(parents=True, exist_ok=True)
        self._append(self.MOVES_CSV, self.MOVE_COLUMNS)

    def _append(self, name: str, row: tuple | list) -> None:
        with (self.directory / name).open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def save_calibration(self, cal: Calibration) -> None:
        cal.save_json(self.directory)

    def frame(self, image: np.ndarray, timestamp: float, phase: str) -> None:
        if not self.record_frames:
            return
        name = f"{self._frame_no:06d}.jpg"
        cv2.imwrite(str(self.directory / "frames" / name), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self._append(self.FRAMES_CSV, (name, f"{timestamp:.3f}", phase))
        self._frame_no += 1

    def move(self, log: MoveLog) -> None:
        self._append(
            self.MOVES_CSV,
            (
                log.ply,
                log.uci,
                log.san,
                log.tier,
                f"{log.margin:.0f}",
                f"{log.score:.0f}",
                f"{log.unexplained:.2f}",
                f"{log.timestamp:.3f}",
            ),
        )
