"""Tests for ArUco-marker based board corner detector."""

from __future__ import annotations

import cv2
import numpy as np

from src.corners import BoardDetector, CornerDetector


def _make_frame_with_markers(
    width: int = 800,
    height: int = 600,
    marker_size: int = 80,
    inset: int = 20,
    dictionary_name: str = "DICT_4X4_50",
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    """Render 4 ArUco markers on a blank frame in the standard corner positions.

    Returns (frame, expected_inner_corner_per_id).
    """
    dict_id = getattr(cv2.aruco, dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    frame = np.full((height, width, 3), 255, dtype=np.uint8)

    # (top-left x, top-left y) of each marker in the frame
    positions = {
        0: (inset, inset),                                          # TL marker
        1: (width - inset - marker_size, inset),                    # TR marker
        2: (width - inset - marker_size, height - inset - marker_size),  # BR marker
        3: (inset, height - inset - marker_size),                   # BL marker
    }

    expected: dict[int, tuple[int, int]] = {}
    for marker_id, (x, y) in positions.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        frame[y : y + marker_size, x : x + marker_size] = marker_bgr

        # Inner corner of each marker = the one pointing into the board
        inner = {
            0: (x + marker_size - 1, y + marker_size - 1),  # TL marker -> bottom-right
            1: (x, y + marker_size - 1),                    # TR marker -> bottom-left
            2: (x, y),                                       # BR marker -> top-left
            3: (x + marker_size - 1, y),                    # BL marker -> top-right
        }[marker_id]
        expected[marker_id] = inner

    return frame, expected


def test_corner_detector_alias():
    """CornerDetector is an alias for BoardDetector for backwards compatibility."""
    assert CornerDetector is BoardDetector


def test_detect_returns_four_corners_with_all_markers_present():
    frame, expected_inner = _make_frame_with_markers()
    detector = BoardDetector(min_area_px=10_000)
    corners = detector.detect(frame)
    assert corners is not None
    assert corners.shape == (4, 2)
    assert corners.dtype == np.float32

    # Output is in TL, TR, BR, BL order
    for output_idx, marker_id in enumerate((0, 1, 2, 3)):
        ex, ey = expected_inner[marker_id]
        cx, cy = corners[output_idx]
        assert abs(cx - ex) <= 2.0, f"corner {output_idx} x off: got {cx}, expected {ex}"
        assert abs(cy - ey) <= 2.0, f"corner {output_idx} y off: got {cy}, expected {ey}"


def test_detect_returns_none_when_any_marker_missing():
    frame, _ = _make_frame_with_markers()
    # Erase the BR marker region with white pixels
    frame[450:560, 700:780] = 255
    detector = BoardDetector(min_area_px=10_000)
    assert detector.detect(frame) is None


def test_detect_returns_none_when_no_markers():
    frame = np.full((600, 800, 3), 255, dtype=np.uint8)
    detector = BoardDetector(min_area_px=10_000)
    assert detector.detect(frame) is None


def test_detect_returns_none_when_area_below_threshold():
    frame, _ = _make_frame_with_markers(width=800, height=600, marker_size=30, inset=300)
    detector = BoardDetector(min_area_px=1_000_000)
    assert detector.detect(frame) is None


def test_unknown_dictionary_raises():
    try:
        BoardDetector(dictionary_name="DICT_DOES_NOT_EXIST")
    except ValueError as exc:
        assert "Unknown ArUco dictionary" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown dictionary")
