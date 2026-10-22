"""Opening theory: name the opening a game is in and list the book moves from the current position.

The book is chessreader/data/openings.tsv (columns eco, name, pgn; one named line per row). Every
position along every line is a book position; the position at the end of a line carries its name.
Positions are matched by EPD (placement, side to move, castling, legal en passant), so a game that
reaches a book position by another move order is recognised.
"""

from __future__ import annotations

import csv
import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path

import chess

from .settings import OpeningsSettings, resolve_path

logger = logging.getLogger(__name__)

DEFAULT_BOOK = Path(__file__).resolve().parent / "data" / "openings.tsv"
_STARTING_EPD = chess.Board().epd()


@dataclass
class OpeningInfo:
    eco: str | None  # ECO of the deepest named position the game has reached so far
    name: str | None  # its name
    in_book: bool  # the current position is a known book position
    book_moves: list[chess.Move] = field(default_factory=list)  # theory continuations, most mainline first
    left_book_ply: int | None = None  # ply (1-based) of the move that left the book; None while in book


def _parse_line(pgn: str) -> list[str]:
    """SAN tokens of '1. e4 e5 2. Nf3' (move numbers like '1.' or '1...' dropped)."""
    return [tok for tok in pgn.split() if not tok.rstrip(".").isdigit() and not tok.endswith(".")]


class OpeningBook:
    def __init__(self) -> None:
        self._names: dict[str, tuple[str, str]] = {}  # epd -> (eco, name)
        self._weight: dict[str, int] = {}  # epd -> number of book lines through it
        self.lines = 0

    @classmethod
    def load(cls, path: str | Path | None = None) -> OpeningBook:
        book = cls()
        path = Path(path) if path is not None else DEFAULT_BOOK
        with path.open(encoding="utf-8", newline="") as fh:
            for row_no, row in enumerate(csv.DictReader(fh, delimiter="\t"), start=2):
                eco, name, pgn = (row.get("eco") or "").strip(), (row.get("name") or "").strip(), row.get("pgn") or ""
                if not eco or not name:
                    raise ValueError(f"{path}:{row_no}: missing eco or name")
                book._add_line(eco, name, _parse_line(pgn), f"{path}:{row_no}")
        return book

    def _add_line(self, eco: str, name: str, sans: list[str], where: str) -> None:
        board = chess.Board()
        seen = {board.epd()}
        for san in sans:
            try:
                board.push_san(san)
            except ValueError as exc:
                raise ValueError(f"{where}: illegal move {san!r}: {exc}") from None
            seen.add(board.epd())
        for epd in seen:
            self._weight[epd] = self._weight.get(epd, 0) + 1
        self._names.setdefault(board.epd(), (eco, name))  # the first line naming a position wins
        self.lines += 1

    @property
    def positions(self) -> int:
        return len(self._weight)

    def _book_moves(self, board: chess.Board) -> list[chess.Move]:
        scored = []
        for move in board.legal_moves:
            board.push(move)
            weight = self._weight.get(board.epd())
            board.pop()
            if weight:
                scored.append((weight, move))
        scored.sort(key=lambda wm: (-wm[0], wm[1].uci()))
        return [m for _, m in scored]

    def lookup(self, board: chess.Board) -> OpeningInfo:
        """Opening of the game in `board`, replaying board.move_stack from its root position."""
        root = board.root()
        if root.epd() != _STARTING_EPD:
            return OpeningInfo(eco=None, name=None, in_book=False, book_moves=[], left_book_ply=None)
        eco = name = None
        left_book_ply = None
        replay = root
        for ply, move in enumerate(board.move_stack, start=1):
            replay.push(move)
            epd = replay.epd()
            if epd in self._weight:
                left_book_ply = None
                named = self._names.get(epd)
                if named:
                    eco, name = named
            elif left_book_ply is None:
                left_book_ply = ply
        in_book = left_book_ply is None
        return OpeningInfo(
            eco=eco, name=name, in_book=in_book, book_moves=self._book_moves(replay), left_book_ply=left_book_ply
        )

    def book_san(self, board: chess.Board, limit: int = 3) -> list[str]:
        """SAN of up to `limit` book moves from the current position (empty for a custom start)."""
        if board.root().epd() != _STARTING_EPD:
            return []
        return [board.san(m) for m in self._book_moves(board.copy(stack=False))[: max(0, limit)]]


@functools.cache
def default_book() -> OpeningBook:
    """The bundled book, loaded once per process. Raises if the file is missing or broken."""
    return OpeningBook.load()


@functools.cache
def load_book(path: str = "") -> OpeningBook | None:
    """The book at `path` ("" = the bundled one), loaded once; None, warned once, if it can't be read.

    The Board window and the PGN tags both use this, so a broken book only switches openings off.
    """
    try:
        return OpeningBook.load(path) if path else default_book()
    except Exception as exc:
        logger.warning("Opening book not loaded (%s); no opening names or ECO/Opening tags", exc)
        return None


def book_for(settings: OpeningsSettings) -> OpeningBook | None:
    """The book `openings` in config.yaml asks for, or None when openings are off or it can't be read."""
    if not settings.enabled:
        return None
    return load_book(str(resolve_path(settings.book)) if settings.book else "")
