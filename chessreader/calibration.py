"""Board calibration: locate the board once (on an empty board) and map it to a top-down view.

Auto: cv2.findChessboardCornersSB finds the 7x7 inner corners; a least-squares homography
over all 49 maps them to the warp grid, and the outer corners come from extrapolating one
square outward through that homography.
Fallback: click the 4 outer corners.

The warp is never mirrored: corners are always ordered clockwise (TL, TR, BR, BL) in the
image, so only a rotation can be wrong, and the starting position fixes that later.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .display import GREEN, RED, YELLOW, draw_grid, draw_hud, draw_points, draw_quad
from .settings import CalibrationSettings, resolve_path
from .stream import FrameGrabber

logger = logging.getLogger(__name__)

INNER = 7  # an 8x8 board has 7x7 inner corners
CALIBRATION_FILE = "calibration.json"
RAW_FILE = "empty_raw.png"
WARP_FILE = "empty_warped.png"
EDGES_FILE = "empty_edges.png"
_SB_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY


class CalibrationError(ValueError):
    pass


def warp_corners(warp_size: int) -> np.ndarray:
    s = float(warp_size)
    return np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)


@dataclass
class Calibration:
    homography: np.ndarray        # 3x3, image -> warp
    outer_corners: np.ndarray     # (4, 2) image coords, in warp order TL, TR, BR, BL
    frame_size: tuple[int, int]   # (w, h) of the frames it was made from
    warp_size: int
    method: str                   # "auto" or "manual"
    reprojection_px: float | None = None
    inner_corners: np.ndarray | None = None  # (49, 2) image coords, auto only
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def square_px(self) -> float:
        return self.warp_size / 8

    def warp(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if (w, h) != tuple(self.frame_size):
            raise CalibrationError(
                f"Frame is {w}x{h} but calibration was made at {self.frame_size[0]}x{self.frame_size[1]}. Recalibrate."
            )
        return cv2.warpPerspective(frame, self.homography, (self.warp_size, self.warp_size))

    def rotated(self, quarter_turns: int) -> "Calibration":
        """Same calibration with the top-down view turned clockwise by 90 deg per quarter turn."""
        s = float(self.warp_size)
        turn = np.array([[0, -1, s], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        h = self.homography.astype(np.float64)
        corners = self.outer_corners.copy()
        for _ in range(quarter_turns % 4):
            h = turn @ h
            corners = np.roll(corners, 1, axis=0)
        return replace(self, homography=h / h[2, 2], outer_corners=corners)

    def to_json(self) -> dict:
        return {
            "homography": self.homography.tolist(),
            "outer_corners": self.outer_corners.tolist(),
            "frame_size": list(self.frame_size),
            "warp_size": self.warp_size,
            "method": self.method,
            "reprojection_px": self.reprojection_px,
            "inner_corners": None if self.inner_corners is None else self.inner_corners.tolist(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Calibration":
        inner = data.get("inner_corners")
        return cls(
            homography=np.array(data["homography"], dtype=np.float64),
            outer_corners=np.array(data["outer_corners"], dtype=np.float32),
            frame_size=tuple(data["frame_size"]),
            warp_size=int(data["warp_size"]),
            method=data["method"],
            reprojection_px=data.get("reprojection_px"),
            inner_corners=None if inner is None else np.array(inner, dtype=np.float32),
            created_at=data.get("created_at", ""),
        )

    def save(self, directory: Path, raw_empty: np.ndarray, canny_low: int, canny_high: int) -> None:
        """Write the JSON plus the empty-board reference images derived from raw_empty."""
        directory.mkdir(parents=True, exist_ok=True)
        warped = self.warp(raw_empty)
        (directory / CALIBRATION_FILE).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        cv2.imwrite(str(directory / RAW_FILE), raw_empty)
        cv2.imwrite(str(directory / WARP_FILE), warped)
        cv2.imwrite(str(directory / EDGES_FILE), edge_map(warped, canny_low, canny_high))
        logger.info("Calibration saved to %s (%s, fit error %s)", directory, self.method,
                    "n/a" if self.reprojection_px is None else f"{self.reprojection_px:.2f}px")

    @classmethod
    def load(cls, directory: Path) -> "Calibration":
        path = directory / CALIBRATION_FILE
        if not path.exists():
            raise FileNotFoundError(f"No calibration at {path}")
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


def load_reference(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(raw empty frame, warped empty board, its edge map) saved with the calibration."""
    images = []
    for name, flag in ((RAW_FILE, cv2.IMREAD_COLOR), (WARP_FILE, cv2.IMREAD_COLOR), (EDGES_FILE, cv2.IMREAD_GRAYSCALE)):
        img = cv2.imread(str(directory / name), flag)
        if img is None:
            raise FileNotFoundError(f"Missing calibration image {directory / name}")
        images.append(img)
    return images[0], images[1], images[2]


def edge_map(warped: np.ndarray, low: int, high: int) -> np.ndarray:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), low, high)


# ---------------------------------------------------------------- auto detection

MAX_GRID_ERROR = 3.0  # % of a square: how far detected corners may sit from a perfect grid


def grid_error(grid: np.ndarray) -> float:
    """Mean distance of the corners from the best-fitting perfect grid, in % of a square."""
    rows, cols = grid.shape[:2]
    r, c = np.mgrid[0:rows, 0:cols]
    ideal = (np.stack([c, r], axis=-1).reshape(-1, 2) * 100).astype(np.float32)
    pts = grid.reshape(-1, 2).astype(np.float32)
    h, _ = cv2.findHomography(pts, ideal, 0)
    if h is None:
        return float("inf")
    projected = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), h).reshape(-1, 2)
    return float(np.linalg.norm(projected - ideal, axis=1).mean())


def find_grid(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """Inner corners for pattern (cols, rows) as a (rows, cols, 2) grid, or None.

    Tries several OpenCV detectors and keeps the most grid-like result: findChessboardCornersSB
    sometimes reports success with scrambled corners when pieces stand on part of the board.
    """
    attempts = []
    for flags in (_SB_FLAGS, 0, cv2.CALIB_CB_NORMALIZE_IMAGE):
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if found:
            attempts.append(corners)
    found, corners = cv2.findChessboardCorners(gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
        attempts.append(cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria))
    best, best_error = None, MAX_GRID_ERROR
    for corners in attempts:
        grid = corners.reshape(pattern[1], pattern[0], 2).astype(np.float32)
        error = grid_error(grid)
        logger.debug("grid %s candidate error %.2f%%", pattern, error)
        if error < best_error:
            best, best_error = grid, error
    return best


def find_inner_corners(gray: np.ndarray) -> np.ndarray | None:
    """7x7 inner corners as a (7, 7, 2) grid, or None."""
    return find_grid(gray, (INNER, INNER))


def _handedness(grid: np.ndarray) -> float:
    """> 0 when grid columns run like warp x and rows like warp y (no mirror)."""
    col_dir = grid[0, 1] - grid[0, 0]
    row_dir = grid[1, 0] - grid[0, 0]
    return float(col_dir[0] * row_dir[1] - col_dir[1] * row_dir[0])


def canonical_grid(grid: np.ndarray) -> np.ndarray:
    """Un-mirror the grid, then rotate it so grid[0, 0] is the corner nearest the image top-left."""
    if _handedness(grid) < 0:
        grid = grid.transpose(1, 0, 2)
    options = [np.rot90(grid, k, axes=(0, 1)) for k in range(4)]
    return np.ascontiguousarray(min(options, key=lambda g: float(g[0, 0].sum())))


def calibration_from_inner_corners(grid: np.ndarray, frame_size: tuple[int, int], warp_size: int) -> Calibration:
    grid = canonical_grid(grid)
    sq = warp_size / 8
    rows, cols = np.mgrid[0:INNER, 0:INNER]
    warp_pts = np.stack([(cols + 1) * sq, (rows + 1) * sq], axis=-1).reshape(-1, 2).astype(np.float32)
    img_pts = grid.reshape(-1, 2)

    h, _ = cv2.findHomography(img_pts, warp_pts, 0)
    if h is None:
        raise CalibrationError("Couldn't fit a homography to the detected corners")
    projected = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), h).reshape(-1, 2)
    error = float(np.linalg.norm(projected - warp_pts, axis=1).mean())
    outer = cv2.perspectiveTransform(warp_corners(warp_size).reshape(-1, 1, 2), np.linalg.inv(h)).reshape(-1, 2)
    return Calibration(h / h[2, 2], outer.astype(np.float32), frame_size, warp_size, "auto", error, img_pts)


def calibration_from_middle_block(grid: np.ndarray, frame_size: tuple[int, int], warp_size: int) -> Calibration:
    """Calibrate from the empty middle 4 ranks of the starting position (a 7x3 inner-corner grid).

    The block spans all 8 files and ranks 3-6, so it pins down the whole board; the side
    white is on gets fixed later from the piece colors.
    """
    rows, cols = grid.shape[:2]
    if _handedness(grid) < 0:  # un-mirror without changing the grid shape
        grid = grid[:, ::-1]
    sq = warp_size / 8
    r_idx, c_idx = np.mgrid[0:rows, 0:cols]
    if cols == 3:   # columns run across ranks 3-6
        x, y = (c_idx + 3) * sq, (r_idx + 1) * sq
    else:           # rows run across ranks 3-6
        x, y = (c_idx + 1) * sq, (r_idx + 3) * sq
    warp_pts = np.stack([x, y], axis=-1).reshape(-1, 2).astype(np.float32)
    img_pts = np.ascontiguousarray(grid).reshape(-1, 2)
    h, _ = cv2.findHomography(img_pts, warp_pts, 0)
    if h is None:
        raise CalibrationError("Couldn't fit a homography to the middle of the board")
    projected = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), h).reshape(-1, 2)
    error = float(np.linalg.norm(projected - warp_pts, axis=1).mean())
    outer = cv2.perspectiveTransform(warp_corners(warp_size).reshape(-1, 1, 2), np.linalg.inv(h)).reshape(-1, 2)
    outer = outer.astype(np.float32)
    # keep warp (0, 0) at the corner nearest the image top-left, like the full-board path
    turns = int(np.argmin(outer.sum(axis=1)))
    cal = Calibration(h / h[2, 2], outer, frame_size, warp_size, "auto-start", error, img_pts)
    return cal.rotated((4 - turns) % 4)


def detect_board(frame: np.ndarray, warp_size: int) -> Calibration | None:
    """Find the board: all 49 inner corners on an empty board, else the empty middle of the start position."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]
    grid = find_inner_corners(gray)
    if grid is not None:
        return calibration_from_inner_corners(grid, (w, h), warp_size)
    for pattern in ((3, INNER), (INNER, 3)):
        block = find_grid(gray, pattern)
        if block is not None:
            logger.info("Board found from the empty middle ranks (starting position)")
            return calibration_from_middle_block(block, (w, h), warp_size)
    return None


# ---------------------------------------------------------------- manual corners

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points clockwise in image coords (y down), starting top-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def calibration_from_outer_corners(pts: np.ndarray, frame_size: tuple[int, int], warp_size: int) -> Calibration:
    corners = order_corners(pts)
    if not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
        raise CalibrationError("The 4 corners don't form a convex shape. Click the board's outer corners.")
    if cv2.contourArea(corners) < 0.02 * frame_size[0] * frame_size[1]:
        raise CalibrationError("The board outline is too small. Click the board's outer corners.")
    h = cv2.getPerspectiveTransform(corners, warp_corners(warp_size))
    return Calibration(h / h[2, 2], corners, frame_size, warp_size, "manual")


def click_outer_corners(frame: np.ndarray, window: str = "Click 4 corners") -> np.ndarray | None:
    """Let the user click the 4 outer board corners. Returns (4, 2) or None if cancelled."""
    points: list[tuple[int, int]] = []
    cursor = [0, 0]

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        cursor[0], cursor[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            vis = frame.copy()
            if points:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(vis, [pts.reshape(-1, 1, 2)], len(points) == 4, GREEN, 2, cv2.LINE_AA)
                draw_points(vis, pts.astype(np.float32), GREEN, radius=4)
            _draw_magnifier(vis, frame, cursor)
            done = len(points) == 4
            draw_hud(vis, [
                f"Click the 4 OUTER corners of the playing area ({len(points)}/4)",
                "ENTER accept   U undo   R reset   ESC cancel" if done else "U undo   R reset   ESC cancel",
            ])
            cv2.imshow(window, vis)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (8, ord("u")) and points:
                points.pop()
            elif key == ord("r"):
                points.clear()
            elif key in (13, 10) and done:
                return np.array(points, dtype=np.float32)
    finally:
        cv2.destroyWindow(window)


def _draw_magnifier(vis: np.ndarray, frame: np.ndarray, cursor: list[int], radius: int = 20, zoom: int = 5) -> None:
    """4x-ish zoom of the area under the cursor in the top-right corner, for precise clicks."""
    h, w = frame.shape[:2]
    x, y = cursor
    x0, y0 = max(0, x - radius), max(0, y - radius)
    x1, y1 = min(w, x + radius), min(h, y + radius)
    if x1 <= x0 or y1 <= y0:
        return
    patch = cv2.resize(frame[y0:y1, x0:x1], ((x1 - x0) * zoom, (y1 - y0) * zoom), interpolation=cv2.INTER_NEAREST)
    ph, pw = patch.shape[:2]
    if ph >= h or pw >= w:
        return
    cx, cy = (x - x0) * zoom, (y - y0) * zoom
    cv2.line(patch, (cx, 0), (cx, ph), RED, 1)
    cv2.line(patch, (0, cy), (pw, cy), RED, 1)
    vis[0:ph, w - pw:w] = patch
    cv2.rectangle(vis, (w - pw, 0), (w - 1, ph - 1), (255, 255, 255), 1)


# ---------------------------------------------------------------- interactive flow

WINDOW = "Calibration"
PREVIEW_WINDOW = "Calibration preview"


def _auto_detect(grabber: FrameGrabber, settings: CalibrationSettings, after: int) -> tuple[Calibration | None, np.ndarray | None, str]:
    best_error = None
    for _ in range(settings.detect_attempts):
        frame = grabber.wait_for_frame(after, timeout=15.0)  # this camera can take several seconds per frame
        if frame is None:
            return None, None, "No frames from the camera."
        after = frame.index
        t0 = time.perf_counter()
        cal = detect_board(frame.image, settings.warp_size)
        logger.debug("findChessboardCornersSB took %.0f ms, found=%s", (time.perf_counter() - t0) * 1000, cal is not None)
        if cal is None:
            continue
        if cal.reprojection_px <= settings.max_reprojection_px:
            return cal, frame.image, ""
        best_error = cal.reprojection_px if best_error is None else min(best_error, cal.reprojection_px)
        logger.info("Board found but fit error %.2fpx > %.2fpx", cal.reprojection_px, settings.max_reprojection_px)
    if best_error is not None:
        return None, None, f"Board found but fit error {best_error:.1f}px is too high (lens distortion?). Press M."
    return None, None, "Board not found. Empty it, show all of it, light it evenly, or press M."


def _confirm(cal: Calibration, frame: np.ndarray, show_labels: bool) -> bool:
    vis = frame.copy()
    if cal.inner_corners is not None:
        draw_points(vis, cal.inner_corners, YELLOW)
    draw_quad(vis, cal.outer_corners)
    fit = "" if cal.reprojection_px is None else f", fit error {cal.reprojection_px:.2f}px"
    draw_hud(vis, [f"{cal.method} calibration{fit}", "ENTER accept & save   R retry"])
    warped = cal.warp(frame)
    draw_grid(warped, show_labels)
    draw_hud(warped, ["Grid lines should sit on the square edges"], origin=(10, warped.shape[0] - 14))

    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            cv2.imshow(WINDOW, vis)
            cv2.imshow(PREVIEW_WINDOW, warped)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):
                return True
            if key in (ord("r"), 27):
                return False
    finally:
        cv2.destroyWindow(PREVIEW_WINDOW)


def auto_calibrate(grabber: FrameGrabber, settings: CalibrationSettings, attempts: int = 5) -> Calibration | None:
    """Calibrate without a window (empty board or starting position); saves on success."""
    cal, frame, error = _auto_detect(grabber, replace_attempts(settings, attempts), -1)
    if cal is None:
        logger.info("Automatic calibration failed: %s", error)
        return None
    cal.save(resolve_path(settings.dir), frame, settings.canny_low, settings.canny_high)
    return cal


def replace_attempts(settings: CalibrationSettings, attempts: int) -> CalibrationSettings:
    return settings.model_copy(update={"detect_attempts": attempts})


def run_calibration(grabber: FrameGrabber, settings: CalibrationSettings, show_labels: bool = True) -> Calibration | None:
    """Interactive calibration on an empty board. Saves and returns the result, or None if cancelled."""
    directory = resolve_path(settings.dir)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    last_index = -1
    frame: np.ndarray | None = None
    message, message_until = "", 0.0
    try:
        while True:
            latest = grabber.wait_for_frame(last_index, timeout=0.5)
            if latest is not None:
                last_index, frame = latest.index, latest.image
            if frame is None:
                blank = np.zeros((480, 640, 3), np.uint8)
                draw_hud(blank, ["Waiting for camera...", "ESC cancel"])
                cv2.imshow(WINDOW, blank)
                if (cv2.waitKey(30) & 0xFF) == 27:
                    return None
                continue

            vis = frame.copy()
            lines = ["CALIBRATION: empty board, or pieces in the starting position",
                     "SPACE auto-detect   M click 4 corners   ESC cancel"]
            if time.monotonic() < message_until:
                lines.append(message)
            draw_hud(vis, lines)
            cv2.imshow(WINDOW, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return None

            candidate, source = None, None
            if key == ord(" "):
                draw_hud(vis, ["", "", "Searching for the board..."], color=YELLOW)
                cv2.imshow(WINDOW, vis)
                cv2.waitKey(1)
                candidate, source, error = _auto_detect(grabber, settings, last_index)
                if candidate is None:
                    message, message_until = error, time.monotonic() + 5
            elif key == ord("m"):
                source = frame.copy()
                pts = click_outer_corners(source)
                if pts is not None:
                    h, w = source.shape[:2]
                    try:
                        candidate = calibration_from_outer_corners(pts, (w, h), settings.warp_size)
                    except CalibrationError as exc:
                        message, message_until = str(exc), time.monotonic() + 5

            if candidate is not None and source is not None and _confirm(candidate, source, show_labels):
                candidate.save(directory, source, settings.canny_low, settings.canny_high)
                return candidate
    finally:
        cv2.destroyWindow(WINDOW)
