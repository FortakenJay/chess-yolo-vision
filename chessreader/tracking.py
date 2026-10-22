"""Follows the camera when it shifts, so a nudged or swaying camera doesn't need recalibrating.

The board is a plane, so any camera shift moves its image by a homography. ORB features in and
around the board (piece tops, printed coordinates, the board's edge, the table) are matched
between the last accepted frame and the current one, and RANSAC fits that homography. The
checker pattern itself repeats every square, so ambiguous matches are dropped (ratio test) or
else a fit could snap one square off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Shift:
    matrix: np.ndarray  # 3x3, previous image -> current image
    inliers: int
    moved_px: float  # mean displacement of the board corners


def board_mask(shape: tuple[int, int], corners: np.ndarray, margin: float = 0.15) -> np.ndarray:
    """The board quad grown by `margin` of its size, so the border and the table edge count too."""
    h, w = shape
    center = corners.mean(axis=0)
    grown = center + (corners - center) * (1.0 + 2 * margin)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(grown).astype(np.int32), 255)
    return mask


class CameraTracker:
    def __init__(
        self, n_features: int = 3000, ratio: float = 0.75, min_inliers: int = 40, ransac_px: float = 3.0
    ) -> None:
        self.orb = cv2.ORB.create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio = ratio
        self.min_inliers = min_inliers
        self.ransac_px = ransac_px

    def estimate(self, previous: np.ndarray, current: np.ndarray, corners: np.ndarray) -> Shift | None:
        """Homography from `previous` to `current`, or None when it can't be trusted."""
        mask = board_mask(previous.shape[:2], corners)
        g0 = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        k0, d0 = self.orb.detectAndCompute(g0, mask)
        # the board may have moved out of the old mask: search the whole current frame
        k1, d1 = self.orb.detectAndCompute(g1, None)
        if d0 is None or d1 is None or len(k0) < self.min_inliers or len(k1) < self.min_inliers:
            return None
        good = [
            m
            for m, n in (p for p in self.matcher.knnMatch(d0, d1, k=2) if len(p) == 2)
            if m.distance < self.ratio * n.distance
        ]
        if len(good) < self.min_inliers:
            logger.debug("camera tracking: only %d distinctive matches", len(good))
            return None
        src = np.array([k0[m.queryIdx].pt for m in good], np.float32)
        dst = np.array([k1[m.trainIdx].pt for m in good], np.float32)
        found = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_px)
        matrix: np.ndarray | None = found[0]  # None when RANSAC finds no model
        inlier_mask = found[1]
        if matrix is None:
            return None
        inliers = int(inlier_mask.sum())
        if inliers < self.min_inliers:
            logger.debug("camera tracking: %d inliers of %d matches", inliers, len(good))
            return None
        moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float64), matrix).reshape(-1, 2)
        if not cv2.isContourConvex(moved.astype(np.float32)):
            return None
        return Shift(matrix, inliers, float(np.linalg.norm(moved - corners, axis=1).mean()))
