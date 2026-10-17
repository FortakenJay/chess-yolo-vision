"""Synthetic camera frames of a board with pieces, for matcher and gating tests."""

from __future__ import annotations

import chess
import cv2
import numpy as np

from chessreader.calibration import Calibration, calibration_from_outer_corners
from chessreader.geometry import BoardCamera, square_center

FRAME = (800, 600)
# board outer corners as seen by the real ESP32 camera (low side view)
OUTER = np.array([[165, 240], [545, 225], [655, 490], [88, 522]], dtype=np.float32)


def make_camera() -> tuple[Calibration, BoardCamera]:
    cal = calibration_from_outer_corners(OUTER, FRAME, 800)
    return cal, BoardCamera.from_calibration(cal)


def table_texture(seed: int = 1) -> np.ndarray:
    """A wood-grain-like table, so camera tracking has something to hold on to beside the board."""
    w, h = FRAME
    rng = np.random.default_rng(seed)
    table = np.full((h, w, 3), (40, 60, 70), np.uint8)
    for _ in range(400):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        color = tuple(int(c) for c in rng.integers(20, 110, 3))
        cv2.ellipse(table, (x, y), (int(rng.integers(3, 25)), int(rng.integers(2, 10))),
                    float(rng.uniform(0, 180)), 0, 360, color, -1)
    return table


def render(cam: BoardCamera, cal: Calibration, board: chess.Board, rng: np.random.Generator,
           noise: float = 4.0, exposure: float = 0.0, table: np.ndarray | None = None) -> np.ndarray:
    w, h = FRAME
    frame = np.full((h, w, 3), (40, 60, 70), np.uint8) if table is None else table.copy()
    # checkerboard on the table
    top = np.zeros((800, 800, 3), np.uint8)
    for r in range(8):
        for c in range(8):
            top[r * 100:(r + 1) * 100, c * 100:(c + 1) * 100] = (225, 230, 225) if (r + c) % 2 == 0 else (40, 70, 45)
    warped = cv2.warpPerspective(top, np.linalg.inv(cal.homography), (w, h))
    board_mask = cv2.warpPerspective(np.full((800, 800), 255, np.uint8), np.linalg.inv(cal.homography), (w, h)) > 0
    frame[board_mask] = warped[board_mask]
    # pieces, far to near, with a darker outline so neighbours stay distinguishable
    for square, piece in sorted(board.piece_map().items(), key=lambda it: -cam.depth(*square_center(it[0]))):
        poly = cam.silhouette(square, piece.piece_type).round().astype(np.int32).reshape(-1, 1, 2)
        color = (215, 220, 220) if piece.color else (25, 25, 30)
        cv2.fillPoly(frame, [poly], color)
        cv2.polylines(frame, [poly], True, (110, 110, 110), 1)
        # a piece-type cue near the top so promotions differ slightly
        x, y = cam.project(np.array([[*square_center(square), -cam.heights[piece.piece_type]]]))[0]
        cv2.circle(frame, (int(x), int(y) + 4), 2 + piece.piece_type, (128, 128, 128), -1)
    out = frame.astype(np.float32) + exposure + rng.normal(0, noise, frame.shape)
    return np.clip(out, 0, 255).astype(np.uint8)
