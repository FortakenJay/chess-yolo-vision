"""Health checks: is the tracked game still in sync with the real board, and has the board moved?"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import chess
import cv2
import numpy as np

from .reader import BLACK, EMPTY, WHITE

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    mismatches: list[str] = field(default_factory=list)   # e.g. "e4: board says empty, camera sees white"
    checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.mismatches


def compare_occupancy(probs: np.ndarray, board: chess.Board, min_confidence: float = 0.9) -> SyncReport:
    """Squares where the camera is confident and disagrees with the tracked position."""
    names = {EMPTY: "empty", WHITE: "white", BLACK: "black"}
    report = SyncReport()
    for sq in chess.SQUARES:
        row, col = 7 - chess.square_rank(sq), chess.square_file(sq)
        p = probs[row, col]
        if np.isnan(p).any():
            continue
        seen = int(np.argmax(p))
        if p[seen] < min_confidence:
            continue
        report.checked += 1
        piece = board.piece_at(sq)
        expected = EMPTY if piece is None else (WHITE if piece.color else BLACK)
        if seen != expected:
            report.mismatches.append(f"{chess.square_name(sq)}: game says {names[expected]}, camera sees {names[seen]}")
    return report


class SyncMonitor:
    """Flags a desync only when the same squares disagree on consecutive checks."""

    def __init__(self, persist: int = 2) -> None:
        self.persist = persist
        self._history: list[set[str]] = []

    def update(self, report: SyncReport) -> list[str]:
        squares = {m.split(":")[0] for m in report.mismatches}
        self._history = (self._history + [squares])[-self.persist:]
        if len(self._history) < self.persist:
            return []
        persistent = set.intersection(*self._history)
        return sorted(persistent)

    def reset(self) -> None:
        self._history.clear()


class DriftMonitor:
    """Detects the board (or camera) shifting since calibration.

    Compares edges of the current top-down view with the calibration edge map: the share of
    reference edge pixels that still have an edge within a few pixels. Pieces moving change
    a little of that; a bumped board or camera changes nearly all of it.
    """

    def __init__(self, reference_edges: np.ndarray, canny_low: int, canny_high: int,
                 tolerance_px: int = 4, threshold: float = 0.45, persist: int = 3) -> None:
        self.ref = reference_edges > 0
        self.canny = (canny_low, canny_high)
        self.tolerance = tolerance_px
        self.threshold = threshold
        self.persist = persist
        self._low_streak = 0
        self.last_score = 1.0

    def score(self, warped: np.ndarray) -> float:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), *self.canny)
        dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
        if not self.ref.any():
            return 1.0
        return float((dist[self.ref] <= self.tolerance).mean())

    def update(self, warped: np.ndarray) -> bool:
        """True once the board has looked shifted on `persist` consecutive checks."""
        self.last_score = self.score(warped)
        self._low_streak = self._low_streak + 1 if self.last_score < self.threshold else 0
        if self._low_streak == self.persist:
            logger.warning("Board seems to have moved (edge match %.2f < %.2f)", self.last_score, self.threshold)
        return self._low_streak >= self.persist

    def reset(self, reference_edges: np.ndarray | None = None) -> None:
        if reference_edges is not None:
            self.ref = reference_edges > 0
        self._low_streak = 0
