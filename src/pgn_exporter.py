"""PGN export and optional opening lookup."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import chess.pgn
import chess.polyglot

from .types import ExportResult, GameResult

ALLOWED_PGN_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


@dataclass(slots=True)
class PGNMetadata:
    event: str = "Casual Game"
    site: str = "Chess Vision Recorder"
    white: str = "White"
    black: str = "Black"


def _opening_from_book(board: chess.Board, opening_book_path: str | None) -> tuple[str | None, str | None]:
    if not opening_book_path:
        return None, None
    path = Path(opening_book_path)
    if not path.exists():
        return None, None
    try:
        with chess.polyglot.open_reader(str(path)) as reader:
            entry = reader.find(board)
            if not entry:
                return None, None
            return "Polyglot Opening", None
    except Exception:  # noqa: BLE001
        return None, None


def export_game(
    moves: list[chess.Move],
    starting_fen: str,
    result: str,
    exports_dir: str,
    opening_book_path: str | None = None,
    metadata: PGNMetadata | None = None,
) -> ExportResult:
    """Build, save, and validate PGN output."""
    if not moves:
        raise ValueError("Cannot export empty game")
    if result not in ALLOWED_PGN_RESULTS:
        raise ValueError("Invalid PGN result")

    md = metadata or PGNMetadata()
    game = chess.pgn.Game()
    game.headers["Event"] = md.event
    game.headers["Site"] = md.site
    game.headers["Date"] = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    game.headers["White"] = md.white
    game.headers["Black"] = md.black
    game.headers["Result"] = result
    game.headers["FEN"] = starting_fen

    board = chess.Board(starting_fen)
    node = game
    for move in moves:
        node = node.add_variation(move)
        board.push(move)

    opening_name, opening_eco = _opening_from_book(board, opening_book_path)
    if opening_name:
        game.headers["Opening"] = opening_name
    if opening_eco:
        game.headers["ECO"] = opening_eco

    pgn = str(game)
    out_dir = Path(exports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.pgn")
    out_path = out_dir / filename
    out_path.write_text(pgn, encoding="utf-8")
    verify = out_path.read_text(encoding="utf-8")
    if not verify.strip():
        raise ValueError("Written PGN file is empty")

    return ExportResult(pgn=pgn, file_path=str(out_path), opening_name=opening_name, opening_eco=opening_eco)

