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
    moves = [
        "e4",
        "d5",
        "exd5",
        "Qxd5",
        "Nc3",
        "Qa5",
        "d4",
        "c6",
        "Nf3",
        "Bg4",
        "Be2",
        "e6",
        "O-O",
        "Nd7",
        "h3",
        "Bh5",
        "Bf4",
        "Ngf6",
        "Qd2",
        "Bb4",
        "a3",
        "Bxc3",
        "Qxc3",
        "Qxc3",
        "bxc3",
        "O-O",
    ]
    assert_all_correct(play_and_match(setup, chess.Board(), moves))


def test_queenside_castling(setup):
    moves = ["d4", "d5", "Nc3", "Nc6", "Bf4", "Bf5", "Qd2", "Qd7", "O-O-O", "O-O-O"]
    assert_all_correct(play_and_match(setup, chess.Board(), moves, seed=1))


def test_en_passant(setup):
    board = chess.Board()
    for san in ["e4", "a6", "e5", "d5"]:
        board.push_san(san)
    assert_all_correct(play_and_match(setup, board, ["exd6"], seed=2))


def test_promotion_defaults_to_queen(setup):
    board = chess.Board("8/P6k/8/8/8/8/6K1/8 w - - 0 1")
    cal, cam, matcher, reader = setup
    rng = np.random.default_rng(3)
    ref = reader.prepare(render(cam, cal, board, rng))
    after = board.copy()
    after.push(chess.Move.from_uci("a7a8q"))
    result = matcher.match(
        board, reader.change_probability(ref, reader.prepare(render(cam, cal, after, rng)), matcher.region)
    )
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
    result = matcher.match(
        board, reader.change_probability(ref, reader.prepare(render(cam, cal, after, rng)), matcher.region)
    )
    assert result.unexplained > 0.35


def test_faint_black_on_dark_move_still_scores(setup):
    """The live e7-e5: black pawn, dark square to dark square, only ~20% of its outline visibly changed.

    With the default change rate that reads as 'nothing happened'; once the board's measured
    black-on-dark contrast is known, the same evidence favours the move, while plain noise still doesn't.
    """
    _, cam, _, _ = setup
    roi = cam.roi()
    board = chess.Board()
    board.push_san("e4")
    move = chess.Move.from_uci("e7e5")
    rng = np.random.default_rng(3)

    def score(matcher, change):
        return matcher.score_move(board, move, change)[0].score

    plain = MoveMatcher(cam, roi, SCALE)
    tuned = MoveMatcher(cam, roi, SCALE)
    tuned.set_tone_contrast(tuned.q_for_contrast(12.0, 10.0, 28.0), 0.55)  # contrast measured on the real board
    mask = plain.predicted_mask(plain.render(board), board, move)
    faint = np.where(mask & (rng.random(mask.shape) < 0.2), 1.0, 0.0).astype(np.float32)
    noise = (rng.random(mask.shape) < 0.03).astype(np.float32)
    assert score(plain, faint) < 0 < score(tuned, faint)
    assert score(tuned, noise) < 0
