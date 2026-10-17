"""ArUco-marker based board corner detection.

Replaces the previous YOLO segmentation/detection model used to locate the
4 corners of the playing surface.

Setup:
  1. Generate 4 ArUco markers with ``code/generate_aruco_markers.py``.
  2. Print them and tape one to each outer corner of the chessboard:

        ID 0  -> TOP-LEFT
        ID 1  -> TOP-RIGHT
        ID 2  -> BOTTOM-RIGHT
        ID 3  -> BOTTOM-LEFT

  3. Orient every marker the same way (the printed file's "up" is the camera's
     "up") and align the marker's INNER corner (the one nearest the centre
     of the board) exactly with the playing-surface corner.

The detector returns the 4 board corners as ``np.float32`` shape ``(4, 2)``
in TL, TR, BR, BL order, identical to the previous YOLO-based detector so
the rest of the pipeline works unchanged.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# OpenCV's detectMarkers returns each marker's 4 corners in this fixed order:
#   index 0 = top-left
#   index 1 = top-right
#   index 2 = bottom-right
#   index 3 = bottom-left
# For each board corner we pick the marker corner that points INTO the board,
# so the playing surface lines up exactly with the chosen point.
_INNER_CORNER_INDEX: dict[int, int] = {
    0: 2,  # marker at TL of board -> its bottom-right corner
    1: 3,  # marker at TR of board -> its bottom-left corner
    2: 0,  # marker at BR of board -> its top-left corner
    3: 1,  # marker at BL of board -> its top-right corner
}

# Output order matches the rest of the pipeline (TL, TR, BR, BL)
_BOARD_CORNER_ORDER: tuple[int, int, int, int] = (0, 1, 2, 3)


class BoardDetector:
    """Detect 4 board corners using ArUco markers on the playing surface."""

    def __init__(
        self,
        min_area_px: int = 40000,
        dictionary_name: str = "DICT_4X4_50",
    ) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "cv2.aruco is not available. Install opencv-python>=4.7 "
                "(or opencv-contrib-python)."
            )
        dict_id = getattr(cv2.aruco, dictionary_name, None)
        if dict_id is None:
            raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.min_area_px = min_area_px
        logger.info("BoardDetector: ArUco %s", dictionary_name)

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        """Return 4 board corners (TL, TR, BR, BL) or None if not all found."""
        marker_corners, marker_ids, _ = self.detector.detectMarkers(frame)
        if marker_ids is None or len(marker_ids) == 0:
            logger.info("board(aruco): no markers detected")
            return None

        ids = marker_ids.flatten().tolist()
        required = set(_BOARD_CORNER_ORDER)
        missing = required - set(ids)
        if missing:
            logger.info("board(aruco): missing marker IDs %s", sorted(missing))
            return None

        points: list[np.ndarray] = []
        for board_idx in _BOARD_CORNER_ORDER:
            marker_pos = ids.index(board_idx)
            inner_idx = _INNER_CORNER_INDEX[board_idx]
            # marker_corners[i] has shape (1, 4, 2)
            inner_pt = marker_corners[marker_pos][0][inner_idx]
            points.append(inner_pt)
        points_arr = np.array(points, dtype=np.float32)

        height, width = frame.shape[:2]
        if not self._in_bounds(points_arr, width, height):
            logger.info("board(aruco): corner outside frame bounds")
            return None
        if not self._is_convex(points_arr):
            logger.info("board(aruco): corners not convex")
            return None

        area = cv2.contourArea(points_arr.reshape(-1, 1, 2))
        if area < self.min_area_px:
            logger.info("board(aruco): area %.0f < min %d px", area, self.min_area_px)
            return None

        logger.info("board(aruco): OK (area=%.0f px)", area)
        return points_arr

    @staticmethod
    def _in_bounds(points: np.ndarray, width: int, height: int) -> bool:
        return bool(
            np.all(points[:, 0] >= 0)
            and np.all(points[:, 0] < width)
            and np.all(points[:, 1] >= 0)
            and np.all(points[:, 1] < height)
        )

    @staticmethod
    def _is_convex(points: np.ndarray) -> bool:
        return bool(cv2.isContourConvex(points.reshape(-1, 1, 2).astype(np.float32)))


CornerDetector = BoardDetector
