import chess
import pytest

from src.errors import IllegalMoveError
from src.move_recorder import MoveRecorder


def _apply_and_get_fen(board: chess.Board, uci: str) -> str:
    board.push(chess.Move.from_uci(uci))
    return board.fen()


def test_normal_move():
    recorder = MoveRecorder()
    b = chess.Board()
    fen = _apply_and_get_fen(b, "e2e4")
    event = recorder.try_apply_fen(fen)
    assert event is not None
    assert event.move_uci == "e2e4"


def test_castling_move():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    recorder = MoveRecorder(starting_fen=fen)
    b = chess.Board(fen)
    next_fen = _apply_and_get_fen(b, "e1g1")
    event = recorder.try_apply_fen(next_fen)
    assert event is not None
    assert event.move_uci == "e1g1"


def test_en_passant():
    fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    recorder = MoveRecorder(starting_fen=fen)
    b = chess.Board(fen)
    next_fen = _apply_and_get_fen(b, "e5d6")
    event = recorder.try_apply_fen(next_fen)
    assert event is not None
    assert event.move_uci == "e5d6"


def test_promotion():
    fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
    recorder = MoveRecorder(starting_fen=fen, promotion_selector=lambda _: "q")
    b = chess.Board(fen)
    next_fen = _apply_and_get_fen(b, "a7a8q")
    event = recorder.try_apply_fen(next_fen)
    assert event is not None
    assert event.move_uci.endswith("q")


def test_illegal_move():
    recorder = MoveRecorder()
    with pytest.raises(IllegalMoveError):
        recorder.try_apply_fen("8/8/8/8/8/8/8/8 w - - 0 1")

