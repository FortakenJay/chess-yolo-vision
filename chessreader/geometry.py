"""3D camera model of the board, recovered from the calibration homography.

Board coordinates are in squares: X runs along warp x, Y along warp y, Z points into
the table, so a piece of height h has its top at Z = -h. With the board oriented
white-at-bottom, square (file f, rank r) has its center at (f + 0.5, 7 - r + 0.5).

The focal length comes from the homography itself (orthogonality of the board axes,
principal point at the image center), so no checkerboard intrinsics calibration is needed.
Lens distortion is ignored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import chess
import cv2
import numpy as np

from .calibration import Calibration

logger = logging.getLogger(__name__)

# Piece heights in squares (a Staunton set on a 2.25" board) and silhouette radii.
DEFAULT_HEIGHTS = {
    chess.PAWN: 0.85,
    chess.KNIGHT: 1.2,
    chess.BISHOP: 1.4,
    chess.ROOK: 1.0,
    chess.QUEEN: 1.55,
    chess.KING: 1.75,
}
BASE_RADIUS = 0.36
TOP_RADIUS = 0.24


def square_center(square: chess.Square) -> tuple[float, float]:
    return chess.square_file(square) + 0.5, 7 - chess.square_rank(square) + 0.5


@dataclass
class BoardCamera:
    camera_matrix: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    image_size: tuple[int, int]
    heights: dict[int, float]
    base_radius: float = BASE_RADIUS
    top_radius: float = TOP_RADIUS

    @classmethod
    def from_calibration(cls, cal: Calibration, height_scale: float = 1.0, focal: float | None = None) -> "BoardCamera":
        w, h = cal.frame_size
        cx, cy = w / 2.0, h / 2.0
        # board (square units) -> image homography
        to_warp_units = np.diag([1.0 / cal.square_px, 1.0 / cal.square_px, 1.0])
        g = np.linalg.inv(to_warp_units @ cal.homography)
        g = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]]) @ g
        f = focal if focal else _focal_from_homography(g)
        if f is None or not 0.5 * w < f < 2.5 * w:
            # a near-overhead view barely constrains the focal length; use the OV2640's ~66 deg diagonal FOV
            fallback = 0.5 * np.hypot(w, h) / np.tan(np.radians(33))
            logger.info("Focal length from the board is unusable (%s); assuming %.0f px", f, fallback)
            f = fallback
        k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

        obj, img = _calibration_points(cal)
        ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError("solvePnP failed for the board pose")
        heights = {p: v * height_scale for p, v in DEFAULT_HEIGHTS.items()}
        cam = cls(k, rvec, tvec, (w, h), heights)
        logger.info("Board camera: focal %.0f px, camera %.1f squares from board center, %.0f deg elevation",
                    f, cam.distance_to_center(), cam.elevation_deg())
        return cam

    def project(self, points: np.ndarray) -> np.ndarray:
        pts, _ = cv2.projectPoints(np.asarray(points, np.float64).reshape(-1, 3), self.rvec, self.tvec,
                                   self.camera_matrix, None)
        return pts.reshape(-1, 2)

    def depth(self, x: float, y: float, z: float = 0.0) -> float:
        r, _ = cv2.Rodrigues(self.rvec)
        return float((r @ np.array([x, y, z]) + self.tvec.ravel())[2])

    def camera_position(self) -> np.ndarray:
        r, _ = cv2.Rodrigues(self.rvec)
        return (-r.T @ self.tvec).ravel()

    def distance_to_center(self) -> float:
        return float(np.linalg.norm(self.camera_position() - np.array([4.0, 4.0, 0.0])))

    def elevation_deg(self) -> float:
        c = self.camera_position()
        horizontal = np.hypot(c[0] - 4.0, c[1] - 4.0)
        return float(np.degrees(np.arctan2(-c[2], horizontal)))

    def silhouette(self, square: chess.Square, piece_type: int, n: int = 16) -> np.ndarray:
        """Image polygon of a tapered cylinder standing on the square."""
        x, y = square_center(square)
        h = self.heights[piece_type]
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        base = np.stack([x + self.base_radius * np.cos(t), y + self.base_radius * np.sin(t), np.zeros(n)], 1)
        top = np.stack([x + self.top_radius * np.cos(t), y + self.top_radius * np.sin(t), np.full(n, -h)], 1)
        pts = self.project(np.vstack([base, top])).astype(np.float32)
        return cv2.convexHull(pts).reshape(-1, 2)

    def board_polygon(self) -> np.ndarray:
        return self.project(np.array([[0, 0, 0], [8, 0, 0], [8, 8, 0], [0, 8, 0]], np.float64))

    def roi(self, margin: int = 8) -> tuple[int, int, int, int]:
        """Image box (x0, y0, x1, y1) covering the board and the tallest pieces on it."""
        corners = [[x, y, z] for x in (0, 8) for y in (0, 8) for z in (0, -max(self.heights.values()) - 0.2)]
        pts = self.project(np.array(corners, np.float64))
        w, h = self.image_size
        x0, y0 = np.floor(pts.min(0)).astype(int) - margin
        x1, y1 = np.ceil(pts.max(0)).astype(int) + margin
        return max(0, x0), max(0, y0), min(w, x1), min(h, y1)

    def render_labels(self, board: chess.Board, scale: float = 1.0,
                      roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """Label image: 0 = no piece, else 1 + square * 12 + piece index (nearest piece wins).

        With roi=(x0, y0, x1, y1) the image covers only that box, at the given scale.
        """
        w, h = self.image_size
        x0, y0, x1, y1 = roi if roi is not None else (0, 0, w, h)
        out = np.zeros((int(round((y1 - y0) * scale)), int(round((x1 - x0) * scale))), np.int16)
        origin = np.array([x0, y0], np.float32)
        for square, piece in sorted(board.piece_map().items(), key=lambda item: -self._depths[item[0]]):
            poly = np.round((self._silhouette_cached(square, piece.piece_type) - origin) * scale).astype(np.int32)
            code = 1 + square * 12 + (piece.piece_type - 1) + (0 if piece.color else 6)
            cv2.fillPoly(out, [poly.reshape(-1, 1, 2)], int(code))
        return out

    def __post_init__(self) -> None:
        self._sil_cache: dict[tuple[int, int], np.ndarray] = {}
        self._depths = {sq: self.depth(*square_center(sq)) for sq in chess.SQUARES}

    def _silhouette_cached(self, square: chess.Square, piece_type: int) -> np.ndarray:
        key = (square, piece_type)
        if key not in self._sil_cache:
            self._sil_cache[key] = self.silhouette(square, piece_type)
        return self._sil_cache[key]


def orient_from_start(cal: Calibration, frame: np.ndarray, height_scale: float = 1.0,
                      board: chess.Board | None = None) -> tuple[Calibration, BoardCamera, float]:
    """Turn the calibration so white (rank 1) is at the bottom of the top-down view.

    Tries all four quarter turns with the given position (default: the starting position)
    and keeps the one where the pixels predicted to show white pieces are brightest
    relative to the black ones. Returns (calibration, camera, contrast); contrast near 0
    means the position isn't on the board or the pieces aren't visible.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    start = board if board is not None else chess.Board()
    best: tuple[float, Calibration, BoardCamera] | None = None
    for k in range(4):
        turned = cal.rotated(k)
        cam = BoardCamera.from_calibration(turned, height_scale)
        labels = cam.render_labels(start)
        piece_idx = (labels - 1) % 12
        white = (labels > 0) & (piece_idx < 6)
        black = (labels > 0) & (piece_idx >= 6)
        if not white.any() or not black.any():
            continue
        contrast = float(gray[white].mean() - gray[black].mean())
        logger.debug("orientation %d: white-black brightness %.1f", k, contrast)
        if best is None or contrast > best[0]:
            best = (contrast, turned, cam)
    if best is None:
        raise ValueError("The board isn't in view")
    contrast, turned, cam = best
    return turned, cam, contrast


def _focal_from_homography(g: np.ndarray) -> float | None:
    """Focal length from a centered board->image homography (square pixels, no skew)."""
    h1, h2 = g[:, 0], g[:, 1]
    estimates = []
    denom = h1[2] * h2[2]
    if abs(denom) > 1e-12:
        estimates.append(-(h1[0] * h2[0] + h1[1] * h2[1]) / denom)
    denom = h1[2] ** 2 - h2[2] ** 2
    if abs(denom) > 1e-12:
        estimates.append(-((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / denom)
    valid = [e for e in estimates if e > 0]
    return float(np.sqrt(np.mean(valid))) if valid else None


def _calibration_points(cal: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """Board-plane points (squares) and their image positions for solvePnP."""
    inv = np.linalg.inv(cal.homography)
    grid = np.array([[x, y] for x in range(9) for y in range(9)], np.float64)
    warp_pts = (grid * cal.square_px).reshape(-1, 1, 2)
    img = cv2.perspectiveTransform(warp_pts, inv).reshape(-1, 2)
    obj = np.hstack([grid, np.zeros((len(grid), 1))])
    return obj, img
