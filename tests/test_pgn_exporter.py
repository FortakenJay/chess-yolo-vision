from pathlib import Path

import chess
import pytest

from src.pgn_exporter import export_game


def test_valid_pgn_output(tmp_path: Path):
    board = chess.Board()
    moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")]
    for m in moves:
        assert m in board.legal_moves
        board.push(m)

    out = export_game(
        moves=moves,
        starting_fen=chess.STARTING_FEN,
        result="*",
        exports_dir=str(tmp_path),
    )
    assert "[Event" in out.pgn
    assert Path(out.file_path).exists()


def test_empty_game_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        export_game([], chess.STARTING_FEN, "*", str(tmp_path))

