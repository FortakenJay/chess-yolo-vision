import chess
import numpy as np
import pytest

from chessreader.matcher import MoveMatcher
from chessreader.reader import ChangeReader
from tests.scene import make_camera, render

SCALE = 0.5


@pytest.fixture(scope="module")
def setup():
    cal, cam = make_camera()
    roi = cam.roi()
    return cal, cam, MoveMatcher(cam, roi, SCALE), ChangeReader(roi, SCALE)


def play_and_match(setup, board: chess.Board, moves: list[str], seed: int = 0) -> list[tuple[str, str, float]]:
    cal, cam, matcher, reader = setup
    rng = np.random.default_rng(seed)
    ref = reader.prepare(render(cam, cal, board, rng))
    results = []
    for i, san in enumerate(moves):
        move = board.parse_san(san)
        after = board.copy()
        after.push(move)
        # alternate small exposure drifts like the real auto-exposure
        cur = reader.prepare(render(cam, cal, after, rng, exposure=(-6 if i % 2 else 5)))
        result = matcher.match(board, reader.change_probability(ref, cur, matcher.region))
        best = result.best.move
        results.append((move.uci(), best.uci(), result.margin))
        board.push(best)
        ref = cur
    return results


def assert_all_correct(results):
    wrong = [(want, got) for want, got, _ in results if want != got]
    assert not wrong, f"wrong moves: {wrong}"
    assert min(m for _, _, m in results) > 0


def test_opening_with_captures_and_castling(setup):
    moves = "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 c6 Nf3 Bg4 Be2 e6 O-O Nd7 h3 Bh5 Bf4 Ngf6 Qd2 Bb4 a3 Bxc3 Qxc3 Qxc3 bxc3 O-O".split()
    assert_all_correct(play_and_match(setup, chess.Board(), moves))


def test_queenside_castling(setup):
    moves = "d4 d5 Nc3 Nc6 Bf4 Bf5 Qd2 Qd7 O-O-O O-O-O".split()
    assert_all_correct(play_and_match(setup, chess.Board(), moves, seed=1))


def test_en_passant(setup):
    board = chess.Board()
    for san in "e4 a6 e5 d5".split():
        board.push_san(san)
    assert_all_correct(play_and_match(setup, board, ["exd6"], seed=2))


def test_promotion_defaults_to_queen(setup):
    board = chess.Board("8/P6k/8/8/8/8/6K1/8 w - - 0 1")
    cal, cam, matcher, reader = setup
    rng = np.random.default_rng(3)
    ref = reader.prepare(render(cam, cal, board, rng))
    after = board.copy()
    after.push(chess.Move.from_uci("a7a8q"))
    result = matcher.match(board, reader.change_probability(ref, reader.prepare(render(cam, cal, after, rng)), matcher.region))
    assert result.best.move == chess.Move.from_uci("a7a8q")
    assert result.is_promotion


def test_no_change_gives_no_positive_move(setup):
    cal, cam, matcher, reader = setup
    rng = np.random.default_rng(4)
    board = chess.Board()
    a = reader.prepare(render(cam, cal, board, rng))
    b = reader.prepare(render(cam, cal, board, rng, exposure=8))
    result = matcher.match(board, reader.change_probability(a, b, matcher.region))
    assert result.best.score < 0
    assert result.total_change < 500


def test_two_moves_between_reads_leave_unexplained_change(setup):
    """If a move is missed (two moves happen between reads), no single move explains it all."""
    cal, cam, matcher, reader = setup
    rng = np.random.default_rng(6)
    board = chess.Board()
    ref = reader.prepare(render(cam, cal, board, rng))
    after = board.copy()
    after.push_san("e4")
    after.push_san("d5")
    result = matcher.match(board, reader.change_probability(ref, reader.prepare(render(cam, cal, after, rng)), matcher.region))
    assert result.unexplained > 0.35
