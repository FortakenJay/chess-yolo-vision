"""Piece detection and board mapping."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .types import BoardMap, DetectedPiece, Orientation

logger = logging.getLogger(__name__)

PIECE_TO_FEN: dict[str, str] = {
    "white-king": "K",
    "white-queen": "Q",
    "white-rook": "R",
    "white-bishop": "B",
    "white-knight": "N",
    "white-pawn": "P",
    "black-king": "k",
    "black-queen": "q",
    "black-rook": "r",
    "black-bishop": "b",
    "black-knight": "n",
    "black-pawn": "p",
}

ALIAS_LABELS: dict[str, str] = {
    "bishop": "white-bishop",
}


def _square_name(file_idx: int, rank_idx: int) -> str:
    return f"{chr(ord('a') + file_idx)}{8 - rank_idx}"


class PieceDetector:
    """Detect pieces in a warped board and map them to board squares."""

    def __init__(
        self,
        model_path: str,
        conf: float,
        warp_size: int,
        orientation: Orientation = Orientation.OVERHEAD_WHITE_BOTTOM,
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Piece model not found: {model_path}")
        from ultralytics import YOLO  # lazy import for tests

        self.model = YOLO(model_path)
        self.conf = conf
        self.warp_size = warp_size
        self.orientation = orientation

    def detect(self, warped_frame: np.ndarray) -> list[DetectedPiece]:
        if warped_frame.shape[:2] != (self.warp_size, self.warp_size):
            raise ValueError("warped frame dimensions do not match warp_size")
        results = self.model.predict(source=warped_frame, conf=self.conf, verbose=False)
        if not results:
            return []
        result = results[0]
        if result.boxes is None:
            return []

        detections: list[DetectedPiece] = []
        names = result.names
        for i, xyxy in enumerate(result.boxes.xyxy.cpu().numpy()):
            conf = float(result.boxes.conf[i].item())
            class_id = int(result.boxes.cls[i].item())
            raw_label = str(names[class_id])
            label = ALIAS_LABELS.get(raw_label, raw_label)
            if label not in PIECE_TO_FEN:
                logger.warning("Skipping unknown class label: %s", raw_label)
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1 = max(0, min(self.warp_size - 1, x1))
            y1 = max(0, min(self.warp_size - 1, y1))
            x2 = max(0, min(self.warp_size - 1, x2))
            y2 = max(0, min(self.warp_size - 1, y2))
            detections.append(DetectedPiece(label=label, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2))
        return detections

    def _pixel_to_square(self, cx: float, cy: float, cell: int) -> tuple[int, int]:
        """Convert pixel centre coordinates to (rank_idx, file_idx) chess indices.

        rank_idx 0  = chess rank 8 (black's back rank)
        rank_idx 7  = chess rank 1 (white's back rank)
        file_idx 0  = file a, file_idx 7 = file h

        Side-view orientations swap the x/y axes so that the depth axis of the
        board (ranks 1-8) runs left-to-right across the image instead of
        top-to-bottom, equivalent to a 90° rotation of the overhead view.
        """
        col = int(cx // cell)   # raw x grid index (0 = left)
        row = int(cy // cell)   # raw y grid index (0 = top)

        if self.orientation == Orientation.OVERHEAD_WHITE_BOTTOM:
            # Standard: x→file, y→rank (rank 8 at top, rank 1 at bottom)
            return row, col

        if self.orientation == Orientation.OVERHEAD_WHITE_TOP:
            # 180° flip of the above
            return 7 - row, 7 - col

        if self.orientation == Orientation.SIDE_WHITE_LEFT:
            # 90° CW rotation of OVERHEAD_WHITE_BOTTOM:
            #   left of image = rank 1 (white's back rank)
            #   top of image  = file a
            rank_idx = 7 - col   # col 0 (left) → rank_idx 7 → chess rank 1
            file_idx = row       # row 0 (top)  → file_idx 0 → file a
            return rank_idx, file_idx

        # SIDE_WHITE_RIGHT
        # 90° CCW rotation of OVERHEAD_WHITE_BOTTOM:
        #   right of image = rank 1 (white's back rank)
        #   top of image   = file h
        rank_idx = col           # col 7 (right) → rank_idx 7 → chess rank 1
        file_idx = 7 - row       # row 0 (top)   → file_idx 7 → file h
        return rank_idx, file_idx

    def to_board_map(self, detections: list[DetectedPiece]) -> BoardMap | None:
        board_map: BoardMap = {}
        cell = self.warp_size // 8
        for det in detections:
            cx = (det.x1 + det.x2) / 2.0
            cy = (det.y1 + det.y2) / 2.0
            if not (0 <= cx < self.warp_size and 0 <= cy < self.warp_size):
                logger.warning("Skipping out-of-bounds detection for %s", det.label)
                continue
            rank_idx, file_idx = self._pixel_to_square(cx, cy, cell)
            board_map[(rank_idx, file_idx)] = PIECE_TO_FEN[det.label]
            det.square = _square_name(file_idx, rank_idx)

        if not self._plausible(board_map):
            logger.warning("Discarding frame: physically implausible piece counts")
            return None
        return board_map

    @staticmethod
    def _plausible(board_map: BoardMap) -> bool:
        values = list(board_map.values())
        white = [v for v in values if v.isupper()]
        black = [v for v in values if v.islower()]

        if values.count("K") != 1 or values.count("k") != 1:
            return False
        if values.count("P") > 8 or values.count("p") > 8:
            return False
        if len(white) > 16 or len(black) > 16 or len(values) > 32:
            return False
        return True

