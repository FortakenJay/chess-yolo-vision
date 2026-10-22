"""Board window drawing and the R (reset) key, without a camera."""

import chess
import numpy as np
import pytest

from chessreader.display import badge_corner
from chessreader.play import PANEL_HEIGHT, render_board, reset_game
from chessreader.session import GameSession, Phase
from chessreader.settings import Settings
from tests.scene import OVERHEAD, make_camera, render


@pytest.fixture
def setup(tmp_path):
    settings = Settings()
    settings.engine.enabled = False
    settings.storage.exports_dir = str(tmp_path / "exports")
    cal, cam = make_camera(OVERHEAD)
    rng = np.random.default_rng(0)

    def frame(board=None):
        return render(cam, cal, board or chess.Board(), rng)

    def new_session(c):
        return GameSession(settings, c)

    return settings, cal, frame, new_session


def played(session, sans):
    board = chess.Board()
    for san in sans:
        move = session.board.parse_san(san)
        session.record.add_move(move, 1, 0)
        board.push(move)
    return board


def test_render_board_with_and_without_a_question(setup):
    settings, cal, frame, new_session = setup
    session = new_session(cal)
    assert session.start_game(frame()) is None
    board = played(session, ["e4", "e5", "Nf3"])
    image = frame(board)
    size = session.cal.warp_size
    plain = render_board(session, image, settings)
    b = session.board
    session.choices = [b.parse_san("Be7"), b.parse_san("Qe7"), b.parse_san("Nc6")]
    session.phase = Phase.ASK_USER
    asked = render_board(session, image, settings)
    assert plain.shape == asked.shape == (size + PANEL_HEIGHT, size + 26, 3)
    assert not np.array_equal(plain, asked)


def test_badges_sharing_a_square_use_different_corners():
    first = chess.Move.from_uci("f8e7")
    second = chess.Move.from_uci("d8e7")
    a = badge_corner(first)
    b = badge_corner(second, [a])
    assert a != b
    assert a == "bl"  # away from the arrow coming in from the top right


def test_reset_starts_over_and_keeps_the_old_pgn(setup):
    _, cal, frame, new_session = setup
    session = new_session(cal)
    assert session.start_game(frame()) is None
    board = played(session, ["e4"])
    old = session.record

    # pieces still on the wrong squares: new session waits for S
    waiting = reset_game(session, frame(board), new_session)
    assert old.result == "unknown" and '[Result "*"]' in old.path.read_text()
    assert "1. e4" in old.path.read_text()
    assert waiting.phase == Phase.WAIT_START and waiting.record is None

    # pieces put back: R from the waiting session starts the game at once
    playing = reset_game(waiting, frame(), new_session)
    assert playing.phase == Phase.PLAYING and not playing.board.move_stack

    # R in the middle of a question also works
    playing.choices = [chess.Move.from_uci("e2e4")]
    playing.phase = Phase.ASK_USER
    again = reset_game(playing, frame(), new_session)
    assert again.phase == Phase.PLAYING and not again.choices
    assert playing.phase == Phase.ENDED
