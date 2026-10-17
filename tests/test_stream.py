import cv2
import numpy as np
import pytest

from chessreader.stream import FrameGrabber, framesize_code


def test_framesize_lookup():
    assert framesize_code("svga") == (11, (800, 600))
    assert framesize_code("QVGA") == (6, (320, 240))
    with pytest.raises(ValueError):
        framesize_code("4K")


def test_grabber_keeps_newest_frame(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    for i in range(30):
        writer.write(np.full((120, 160, 3), i * 8, np.uint8))
    writer.release()

    grabber = FrameGrabber(str(path), backoff_s=0.05).start()
    try:
        first = grabber.wait_for_frame(timeout=5)
        assert first is not None and first.image.shape == (120, 160, 3)
        later = grabber.wait_for_frame(first.index, timeout=5)
        assert later is not None and later.index > first.index
        assert grabber.latest().index >= later.index
    finally:
        grabber.stop()
