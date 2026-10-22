"""Motion gating: only read the board after it has been still for a while.

Consecutive frames are compared inside the board region (downscaled, blurred grayscale).
A frame counts as "moving" when more than motion_fraction of the pixels changed by more
than pixel_threshold. The board is stable once stable_time_s has passed and at least
min_stable_frames still frames arrived since the last motion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GateState(StrEnum):
    MOVING = "moving"
    SETTLING = "settling"
    STABLE = "stable"


@dataclass
class GateUpdate:
    state: GateState
    motion: float  # fraction of region pixels that changed since the previous frame
    became_stable: bool  # True only on the frame where the board turned stable


class MotionGate:
    def __init__(
        self,
        region_mask: np.ndarray,
        pixel_threshold: float = 18.0,
        motion_fraction: float = 0.004,
        stable_time_s: float = 0.8,
        min_stable_frames: int = 2,
        work_width: int = 240,
    ) -> None:
        h, w = region_mask.shape[:2]
        self.scale = min(1.0, work_width / w)
        size = (max(1, int(round(w * self.scale))), max(1, int(round(h * self.scale))))
        self.size = size
        self.mask = cv2.resize(region_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        self.pixel_threshold = pixel_threshold
        self.motion_fraction = motion_fraction
        self.stable_time_s = stable_time_s
        self.min_stable_frames = min_stable_frames
        self._prev: np.ndarray | None = None
        self._last_motion_t: float | None = None
        self._still_frames = 0
        self._state = GateState.SETTLING

    @property
    def state(self) -> GateState:
        return self._state

    def _prep(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        small = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (5, 5), 0).astype(np.int16)

    def reset(self, timestamp: float) -> None:
        """Forget the stable streak (e.g. after a move was accepted or rejected)."""
        self._last_motion_t = timestamp
        self._still_frames = 0
        self._state = GateState.SETTLING

    def update(self, image: np.ndarray, timestamp: float) -> GateUpdate:
        cur = self._prep(image)
        if self._prev is None:
            self._prev = cur
            self._last_motion_t = timestamp
            return GateUpdate(GateState.SETTLING, 0.0, False)

        motion = float((np.abs(cur - self._prev)[self.mask] > self.pixel_threshold).mean())
        self._prev = cur
        previous = self._state
        if motion > self.motion_fraction:
            self._last_motion_t = timestamp
            self._still_frames = 0
            self._state = GateState.MOVING
        else:
            self._still_frames += 1
            quiet_for = timestamp - (self._last_motion_t if self._last_motion_t is not None else timestamp)
            if self._still_frames >= self.min_stable_frames and quiet_for >= self.stable_time_s:
                self._state = GateState.STABLE
            else:
                self._state = GateState.SETTLING
        became = self._state == GateState.STABLE and previous != GateState.STABLE
        if self._state != previous:
            logger.debug("gate %s -> %s (motion %.4f)", previous.value, self._state.value, motion)
        return GateUpdate(self._state, motion, became)
