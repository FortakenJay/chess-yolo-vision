import numpy as np
import pytest

from src.errors import StreamConnectionError
from src.stream import MJPEGStreamReader


class _FakeCapture:
    def __init__(self, ok=True, frame=None):
        self._ok = ok
        self._frame = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    def isOpened(self):
        return True

    def read(self):
        return self._ok, self._frame

    def release(self):
        return None


def test_invalid_url_rejected():
    with pytest.raises(StreamConnectionError):
        MJPEGStreamReader("foo", 320, 240)


def test_read_valid_frame(monkeypatch):
    monkeypatch.setattr("src.stream.socket.create_connection", lambda *args, **kwargs: _DummySocket())
    monkeypatch.setattr("src.stream.cv2.VideoCapture", lambda *_: _FakeCapture(ok=True))
    reader = MJPEGStreamReader("http://127.0.0.1:81/stream", 320, 240, retries=1)
    frame = reader.read_frame()
    assert frame.shape == (480, 640, 3)


class _DummySocket:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

