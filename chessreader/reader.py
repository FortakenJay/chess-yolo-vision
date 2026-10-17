"""Board observation.

ChangeReader turns two frames (the last accepted position and the new stable frame) into a
per-pixel probability that the scene changed there. The matcher explains that change map
with legal moves, which works even when a low camera angle makes pieces hide the squares
behind them, and it doesn't depend on the piece detector recognizing this particular set.

PieceDetector wraps the YOLO model. On this camera it finds only part of the pieces, so it
is used for promotion piece type only. Swap in a square-crop classifier by implementing
the same detect() signature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Prepared:
    lab: np.ndarray      # float32 Lab of the region of interest, downscaled and blurred
    texture: np.ndarray  # float32 local contrast (std of brightness) at the same scale


def local_contrast(gray: np.ndarray, window: int) -> np.ndarray:
    """Standard deviation of brightness in a window around each pixel.

    A black piece on a dark square has nearly the square's average brightness, but its rim,
    highlights and the shadow line around its base make the patch textured where the empty
    square is flat. That is the cue a person uses to see black on black.
    """
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (window, window))
    sq = cv2.blur(g * g, (window, window))
    return np.sqrt(np.maximum(sq - mean * mean, 0.0))


class ChangeReader:
    def __init__(self, roi: tuple[int, int, int, int], scale: float = 0.5, blur: int = 5,
                 t_low: float = 10.0, t_high: float = 28.0, chroma_weight: float = 1.5,
                 texture_weight: float = 5.0, texture_window: int = 7, texture_tolerance: int = 5) -> None:
        self.roi = roi
        self.scale = scale
        self.blur = blur | 1
        self.t_low = t_low
        self.t_high = t_high
        self.chroma_weight = chroma_weight
        self.texture_weight = texture_weight
        self.texture_window = texture_window | 1
        self.texture_tolerance = texture_tolerance | 1

    def prepare(self, image: np.ndarray) -> Prepared:
        x0, y0, x1, y1 = self.roi
        crop = image[y0:y1, x0:x1]
        size = (int(round((x1 - x0) * self.scale)), int(round((y1 - y0) * self.scale)))
        small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        # texture from the unblurred image: blurring first would erase the cue we're after
        texture = local_contrast(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), self.texture_window)
        small = cv2.GaussianBlur(small, (self.blur, self.blur), 0)
        return Prepared(cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32), texture)

    @staticmethod
    def _match_exposure(ref: np.ndarray, cur: np.ndarray, sel: np.ndarray) -> np.ndarray:
        """Map ref's colors onto cur's exposure: per channel cur ~ gain * ref + offset.

        Auto-exposure and white balance scale bright pixels more than dark ones, so a plain
        offset leaves white pieces looking 'changed'. The fit uses only pixels that look
        unchanged (two passes, dropping outliers), so the moved pieces don't bias it.
        """
        out = ref.copy()
        for c in range(3):
            x, y = ref[..., c][sel], cur[..., c][sel]
            keep = np.ones_like(x, bool)
            gain, offset = 1.0, float(np.median(y - x))
            for _ in range(2):
                if keep.sum() < 50:
                    break
                gain, offset = np.polyfit(x[keep], y[keep], 1)
                keep = np.abs(y - (gain * x + offset)) < 8.0
            clipped = float(np.clip(gain, 0.7, 1.4))
            if clipped != gain:  # a fit thrown off by a hand or a moved piece: keep its offset and every pixel shifts
                gain = clipped
                offset = float(np.median(y[keep] - gain * x[keep])) if keep.sum() >= 50 else float(np.median(y - gain * x))
            out[..., c] = gain * ref[..., c] + offset
        return out

    def difference(self, ref: Prepared, cur: Prepared, mask: np.ndarray | None = None) -> np.ndarray:
        """Per-pixel color distance, after removing any global exposure / white balance change."""
        sel = mask if mask is not None and mask.any() else np.ones(ref.lab.shape[:2], bool)
        delta = cur.lab - self._match_exposure(ref.lab, cur.lab, sel)
        light = np.abs(delta[..., 0])
        chroma = np.hypot(delta[..., 1], delta[..., 2]) * self.chroma_weight
        texture = self._texture_change(ref.texture, cur.texture) * self.texture_weight
        return np.maximum(np.maximum(light, chroma), texture)

    def _texture_change(self, ref: np.ndarray, cur: np.ndarray) -> np.ndarray:
        """How far the current local contrast falls outside the reference's range nearby.

        Square edges are all contrast, so a camera nudge of a pixel would light up every edge if
        pixels were compared one to one. Comparing against the min..max of a small neighbourhood
        ignores shifted edges but still sees a textured piece appear on (or leave) a flat square.
        """
        k = np.ones((self.texture_tolerance, self.texture_tolerance), np.uint8)
        above = cur - cv2.dilate(ref, k)
        below = cv2.erode(ref, k) - cur
        return np.maximum(np.maximum(above, below), 0.0)

    def change_probability(self, ref: Prepared, cur: Prepared, mask: np.ndarray | None = None) -> np.ndarray:
        d = self.difference(ref, cur, mask)
        return np.clip((d - self.t_low) / (self.t_high - self.t_low), 0.0, 1.0).astype(np.float32)


EMPTY, WHITE, BLACK = 0, 1, 2


class OccupancyReader:
    """Per-square empty / white / black probabilities from the top-down view.

    Learns what an empty light square, an empty dark square, a white piece and a black
    piece look like from the starting position, then classifies each square's center
    patch. Squares whose center is hidden behind another piece (low camera angles) are
    reported as unknown (all-NaN). Used for health checks and the debug view; the move
    matcher works from frame differences instead.
    """

    def __init__(self, patch_frac: float = 0.3, std_floor: float = 6.0) -> None:
        self.patch_frac = patch_frac
        self.std_floor = std_floor
        self.templates: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _patches(self, warped: np.ndarray) -> np.ndarray:
        """Mean Lab color of each square's center patch, shape (8, 8, 3), row 0 = rank 8."""
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
        sq = warped.shape[0] / 8
        half = sq * self.patch_frac / 2
        out = np.zeros((8, 8, 3), np.float32)
        for r in range(8):
            for c in range(8):
                cy, cx = (r + 0.5) * sq, (c + 0.5) * sq
                out[r, c] = lab[int(cy - half):int(cy + half), int(cx - half):int(cx + half)].reshape(-1, 3).mean(0)
        return out

    @staticmethod
    def _visible(cam, board, labels: np.ndarray) -> np.ndarray:
        """(8, 8) bool: the square's center isn't covered by a piece standing on another square."""
        import chess

        from .geometry import square_center

        h, w = labels.shape
        vis = np.zeros((8, 8), bool)
        for sq in chess.SQUARES:
            x, y = cam.project(np.array([[*square_center(sq), 0.0]]))[0]
            xi, yi = int(round(x)), int(round(y))
            code = int(labels[yi, xi]) if 0 <= xi < w and 0 <= yi < h else -1
            row, col = 7 - chess.square_rank(sq), chess.square_file(sq)
            vis[row, col] = code == 0 or (code > 0 and (code - 1) // 12 == sq)
        return vis

    def fit(self, warped: np.ndarray, cam, board) -> None:
        """Learn the four looks from the starting position."""
        import chess

        patches = self._patches(warped)
        visible = self._visible(cam, board, cam.render_labels(board))
        groups: dict[str, list[np.ndarray]] = {"light": [], "dark": [], "white": [], "black": []}
        for sq in chess.SQUARES:
            row, col = 7 - chess.square_rank(sq), chess.square_file(sq)
            if not visible[row, col]:
                continue
            piece = board.piece_at(sq)
            if piece is None:
                groups["light" if (row + col) % 2 == 0 else "dark"].append(patches[row, col])
            else:
                groups["white" if piece.color else "black"].append(patches[row, col])
        for name, values in groups.items():
            if len(values) >= 2:
                arr = np.array(values)
                self.templates[name] = (arr.mean(0), np.maximum(arr.std(0), self.std_floor))
        logger.info("Occupancy templates learned: %s", {k: len(v) for k, v in groups.items()})

    def read(self, warped: np.ndarray, cam, board) -> np.ndarray:
        """(8, 8, 3) probabilities [empty, white, black]; NaN where the square is hidden."""
        probs = np.full((8, 8, 3), np.nan, np.float32)
        if not {"light", "dark", "white", "black"} <= self.templates.keys():
            return probs
        patches = self._patches(warped)
        visible = self._visible(cam, board, cam.render_labels(board))

        def nll(x: np.ndarray, name: str) -> float:
            mean, std = self.templates[name]
            return float((((x - mean) / std) ** 2).sum() / 2)

        for r in range(8):
            for c in range(8):
                if not visible[r, c]:
                    continue
                x = patches[r, c]
                e = nll(x, "light" if (r + c) % 2 == 0 else "dark")
                scores = -np.array([e, nll(x, "white"), nll(x, "black")])
                scores -= scores.max()
                p = np.exp(scores)
                probs[r, c] = p / p.sum()
        return probs


@dataclass
class Detection:
    label: str          # e.g. "white-queen", or "bishop" when the model gives no color
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in image pixels

    @property
    def base_point(self) -> tuple[float, float]:
        """Bottom-center of the box: where the piece stands, not where its body is."""
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2, y2


class PieceDetector:
    def __init__(self, weights: str | Path, conf: float = 0.15, imgsz: int = 800, device: str | None = None) -> None:
        from ultralytics import YOLO  # slow import; only when piece types are needed

        self.model = YOLO(str(weights))
        self.names = {int(k): str(v) for k, v in self.model.names.items()}
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        logger.info("Piece model %s: %d classes", weights, len(self.names))

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        out = []
        for cls, conf, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
            out.append(Detection(self.names[int(cls)], float(conf), tuple(box)))
        return out
