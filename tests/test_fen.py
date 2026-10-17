import pytest

from src.errors import FENBuildError
from src.fen import build_fen


def test_build_starting_position_fen():
    board_map = {}
    for i, p in enumerate("rnbqkbnr"):
        board_map[(0, i)] = p
        board_map[(7, i)] = p.upper()
    for i in range(8):
        board_map[(1, i)] = "p"
        board_map[(6, i)] = "P"
    fen = build_fen(board_map)
    assert fen.startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")


def test_build_empty_board():
    fen = build_fen({})
    assert fen.startswith("8/8/8/8/8/8/8/8")


def test_build_single_kings():
    fen = build_fen({(0, 4): "k", (7, 4): "K"})
    assert fen.startswith("4k3/8/8/8/8/8/8/4K3")


def test_invalid_piece_raises():
    with pytest.raises(FENBuildError):
        build_fen({(0, 0): "X"})

