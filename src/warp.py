"""Perspective transformation utilities."""

from __future__ import annotations

import numpy as np
import cv2


def warp_board(frame: np.ndarray, corners: np.ndarray, warp_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Warp board ROI into a square top-down image."""
    if corners.shape != (4, 2):
        raise ValueError("corners must be shape (4, 2)")
    if corners.dtype != np.float32:
        raise ValueError("corners must be float32")

    dst = np.array(
        [
            [0, 0],
            [warp_size - 1, 0],
            [warp_size - 1, warp_size - 1],
            [0, warp_size - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, dst)
    if abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError("perspective matrix is not invertible")

    warped = cv2.warpPerspective(frame, matrix, (warp_size, warp_size))
    if warped.shape[:2] != (warp_size, warp_size):
        raise ValueError("warped output has incorrect dimensions")
    if np.mean(warped) <= 1.0:
        raise ValueError("warped output appears invalid (all black)")

    return warped, matrix

