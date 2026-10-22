"""The live loop (LiveGame) with a fake camera and scripted keys: no windows, no ESP32."""

import chess
import numpy as np
import pytest

from chessreader import play
from chessreader.session import Phase
from chessreader.settings import Settings
from chessreader.stream import Frame
from tests.scene import OVERHEAD, make_camera, render


class FakeGrabber:
    """Serves a fixed list of frames, one per call, then nothing (like a camera that went quiet)."""

    fps = 1.0
    connected = True

    def __init__(self, images: list[np.ndarray]) -> None:
        self.frames = [Frame(i, i * 1.2, img) for i, img in enumerate(images)]

    def wait_for_frame(self, after_index: int = -1, timeout: float = 5.0) -> Frame | None:
        return next((f for f in self.frames if f.index > after_index), None)

    def latest(self) -> Frame | None:
        return self.frames[-1]


@pytest.fixture
def fake_windows(monkeypatch):
    """Replace OpenCV's window calls; waitKey plays back the keys a test queues up."""
    keys: list[str] = []
    for name in ("namedWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(play.cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(play.cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(play.cv2, "waitKey", lambda _ms: ord(keys.pop(0)) if keys else 255)
    return keys


@pytest.fixture
def game(tmp_path):
    settings = Settings()
    settings.engine.enabled = False
    settings.openings.enabled = False
    settings.storage.exports_dir = str(tmp_path / "exports")
    settings.storage.sessions_dir = str(tmp_path / "sessions")
    cal, cam = make_camera(OVERHEAD)
    rng = np.random.default_rng(0)

    def frames(board: chess.Board, n: int) -> list[np.ndarray]:
        return [render(cam, cal, board, rng) for _ in range(n)]

    return settings, cal, frames


def test_keys_start_reset_and_quit(game, fake_windows):
    settings, cal, frames = game
    start = chess.Board()
    grabber = FakeGrabber(frames(start, 4))
    live = play.LiveGame(settings, grabber, cal, grabber.frames[0].image, auto_start=False)
    assert live.session.phase == Phase.WAIT_START
    # S starts the game; R resets it (start position still up, so the new game starts at once); Q quits
    fake_windows.extend(["s", "r", "q"])
    assert live.run() == 0
    assert live.session.phase == Phase.PLAYING
    assert live.session.message.startswith("Game reset.")


def test_unknown_and_out_of_phase_keys_are_ignored(game, fake_windows):
    settings, cal, frames = game
    grabber = FakeGrabber(frames(chess.Board(), 3))
    live = play.LiveGame(settings, grabber, cal, grabber.frames[0].image, auto_start=False)
    # N (new game) only works after a game ended, G/A/E/T need a game: before S they do nothing
    fake_windows.extend(["n", "g", "a", "k", "q"])
    assert live.run() == 0
    assert live.session.phase == Phase.WAIT_START


def test_keys_act_on_the_new_game_after_a_reset(game, fake_windows):
    """Key handlers look up the current session each time, so they follow R (and N) to the new game."""
    settings, cal, frames = game
    grabber = FakeGrabber(frames(chess.Board(), 4))
    live = play.LiveGame(settings, grabber, cal, grabber.frames[0].image, auto_start=False)
    first = live.session
    fake_windows.extend(["s", "r", "x", "q"])
    live.run()
    assert live.session is not first
    assert live.session.message == "OK, using the board as it is now."
