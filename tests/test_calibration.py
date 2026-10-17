import cv2
import numpy as np
import pytest

from chessreader.calibration import (
    Calibration,
    CalibrationError,
    calibration_from_outer_corners,
    canonical_grid,
    detect_board,
    load_reference,
    order_corners,
    _handedness,
)

SQ = 60
MARGIN = 40
WARP = 800
FRAME = (800, 600)
# where the board's outer corners land in the camera frame (TL, TR, BR, BL)
TRUE_OUTER = np.array([[210, 90], [600, 120], [660, 520], [150, 470]], dtype=np.float32)


def render_board() -> tuple[np.ndarray, np.ndarray]:
    """Top-down synthetic empty board and its outer corner coords."""
    size = 8 * SQ + 2 * MARGIN
    img = np.full((size, size), 140, np.uint8)
    for r in range(8):
        for c in range(8):
            color = 225 if (r + c) % 2 == 0 else 35
            img[MARGIN + r * SQ:MARGIN + (r + 1) * SQ, MARGIN + c * SQ:MARGIN + (c + 1) * SQ] = color
    lo, hi = MARGIN, MARGIN + 8 * SQ
    outer = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], dtype=np.float32)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), outer


def camera_frame() -> np.ndarray:
    board, outer = render_board()
    h = cv2.getPerspectiveTransform(outer, TRUE_OUTER)
    frame = cv2.warpPerspective(board, h, FRAME, borderValue=(90, 90, 90))
    return cv2.GaussianBlur(frame, (3, 3), 0)


def signed_area(quad: np.ndarray) -> float:
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def test_auto_detect_extrapolates_outer_corners():
    cal = detect_board(camera_frame(), WARP)
    assert cal is not None
    assert cal.reprojection_px < 1.0
    np.testing.assert_allclose(order_corners(cal.outer_corners), TRUE_OUTER, atol=2.0)


def test_auto_detect_never_mirrors():
    # A mirrored camera image must still give clockwise corners (y-down), i.e. an un-mirrored warp.
    for frame in (camera_frame(), cv2.flip(camera_frame(), 1)):
        cal = detect_board(frame, WARP)
        assert cal is not None
        assert signed_area(cal.outer_corners) > 0


def test_canonical_grid_fixes_handedness_and_start():
    rows, cols = np.mgrid[0:7, 0:7]
    grid = np.stack([cols * 10.0 + 100, rows * 10.0 + 50], axis=-1).astype(np.float32)
    for variant in (grid, grid.transpose(1, 0, 2), np.rot90(grid, 1), np.rot90(grid.transpose(1, 0, 2), 3)):
        out = canonical_grid(variant)
        assert _handedness(out) > 0
        np.testing.assert_allclose(out[0, 0], [100, 50])


def test_order_corners_any_input_order():
    rng = np.random.default_rng(0)
    for _ in range(10):
        shuffled = TRUE_OUTER[rng.permutation(4)]
        np.testing.assert_allclose(order_corners(shuffled), TRUE_OUTER)


def test_manual_corners_warp_matches_auto():
    frame = camera_frame()
    manual = calibration_from_outer_corners(TRUE_OUTER, FRAME, WARP)
    auto = detect_board(frame, WARP)
    diff = np.abs(manual.warp(frame).astype(int) - auto.warp(frame).astype(int))
    assert diff.mean() < 6


def test_manual_rejects_bad_quads():
    with pytest.raises(CalibrationError):
        calibration_from_outer_corners(np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32), FRAME, WARP)


def test_rotated_turns_warp_clockwise():
    frame = camera_frame()
    cal = calibration_from_outer_corners(TRUE_OUTER, FRAME, WARP)
    expected = np.rot90(cal.warp(frame), k=-1)
    got = cal.rotated(1).warp(frame)
    assert np.abs(got.astype(int) - expected.astype(int)).mean() < 3
    np.testing.assert_allclose(cal.rotated(4).homography, cal.homography, atol=1e-9)


def test_save_load_roundtrip(tmp_path):
    frame = camera_frame()
    cal = detect_board(frame, WARP)
    cal.save(tmp_path, frame, 50, 150)
    loaded = Calibration.load(tmp_path)
    np.testing.assert_allclose(loaded.homography, cal.homography)
    np.testing.assert_allclose(loaded.outer_corners, cal.outer_corners)
    assert loaded.frame_size == FRAME and loaded.method == "auto"
    raw, warped, edges = load_reference(tmp_path)
    assert raw.shape == (600, 800, 3) and warped.shape == (WARP, WARP, 3) and edges.shape == (WARP, WARP)


def test_warp_rejects_wrong_frame_size():
    cal = calibration_from_outer_corners(TRUE_OUTER, FRAME, WARP)
    with pytest.raises(CalibrationError):
        cal.warp(np.zeros((240, 320, 3), np.uint8))
