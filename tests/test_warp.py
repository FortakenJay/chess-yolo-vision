import numpy as np
import pytest

from src.warp import warp_board


def test_warp_success():
    frame = np.full((300, 300, 3), 255, dtype=np.uint8)
    corners = np.array([[10, 10], [290, 10], [290, 290], [10, 290]], dtype=np.float32)
    warped, m = warp_board(frame, corners, 200)
    assert warped.shape == (200, 200, 3)
    assert m.shape == (3, 3)


def test_invalid_corner_shape():
    frame = np.full((300, 300, 3), 255, dtype=np.uint8)
    corners = np.array([[10, 10], [290, 10]], dtype=np.float32)
    with pytest.raises(ValueError):
        warp_board(frame, corners, 200)

