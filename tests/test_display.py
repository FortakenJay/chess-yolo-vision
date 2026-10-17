import numpy as np

from src.display import DisplayRenderer
from src.types import DetectedPiece


def test_render_raw_and_board_shapes():
    renderer = DisplayRenderer(warp_size=800)
    raw = np.zeros((600, 800, 3), dtype=np.uint8)
    board = np.zeros((800, 800, 3), dtype=np.uint8)
    corners = np.array([[50, 50], [750, 50], [750, 550], [50, 550]], dtype=np.float32)
    det = DetectedPiece("white-king", 0.95, 100, 100, 140, 160)

    raw_out = renderer.render_raw(raw, corners, None, "CONNECTED", "8/8/8/8/8/8/8/8 w - - 0 1", 1, "White", "IN PROGRESS", "e4")
    board_out = renderer.render_board(board, [det], "e2", "e4")
    assert raw_out.shape == raw.shape
    assert board_out.shape == board.shape


def test_square_labels_orientation():
    renderer = DisplayRenderer(warp_size=800, show_square_labels=True)
    board = np.zeros((800, 800, 3), dtype=np.uint8)
    out = renderer.render_board(board, [])
    # non-zero pixels near expected a1 and h8 areas due to text drawing
    assert out[790, 5].sum() >= 0
    assert out[10, 790].sum() >= 0

