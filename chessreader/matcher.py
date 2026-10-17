"""Legal-move matching: which legal move best explains what changed on the board?

For each legal move, the camera model renders which piece is visible at every pixel before
and after the move (nearer pieces hide farther ones). Pixels whose visible piece differs
form the move's predicted change mask M. With b(p) the observed change probability:

    score(move) = sum over p in M of  b(p) * log(q_in / q_out) + (1 - b(p)) * log((1 - q_in) / (1 - q_out))

which is the log-likelihood ratio of "this move happened" against "nothing happened"
(q_in: chance a pixel inside the mask visibly changes, q_out: noise rate outside it).
Scores are in full-resolution-pixel units so thresholds don't depend on the work scale.

Stockfish is never consulted here: the matcher must report blunders as played.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import chess
import cv2
import numpy as np

from .geometry import BoardCamera

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    move: chess.Move
    score: float
    mask_px: float      # predicted-change area (full-res px)
    explained: float    # observed change inside the mask (full-res px)


@dataclass
class MatchResult:
    candidates: list[Candidate]
    total_change: float          # observed change mass in the board region (full-res px)
    unexplained: float = 0.0     # share of observed change in piece-sized blobs outside the best move's mask
    margin: float = 0.0          # best score minus runner-up score
    is_promotion: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def top(self, n: int = 3) -> list[Candidate]:
        return self.candidates[:n]


class MoveMatcher:
    def __init__(self, cam: BoardCamera, roi: tuple[int, int, int, int], scale: float,
                 q_in: float = 0.55, q_out: float = 0.03) -> None:
        self.cam = cam
        self.roi = roi
        self.scale = scale
        self.reward = math.log(q_in / q_out)
        self.penalty = math.log((1 - q_in) / (1 - q_out))
        self.px = 1.0 / (scale * scale)  # one work pixel in full-res pixels
        self.region = self._region_mask()
        # a blob has to be at least ~1/3 of a square across to count as a piece-sized change
        a, b = cam.project(np.array([[4.0, 4.0, 0.0], [5.0, 4.0, 0.0]], np.float64))
        k = max(3, int(round(np.linalg.norm(b - a) * scale / 3)) | 1)
        self._blob_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def _region_mask(self) -> np.ndarray:
        """Pixels any piece on any square could cover: the only place moves can change anything."""
        board = chess.Board(None)
        for sq in chess.SQUARES:
            board.set_piece_at(sq, chess.Piece(chess.KING, chess.WHITE))
        return self.render(board) > 0

    def unexplained_blobs(self, change: np.ndarray, mask: np.ndarray) -> float:
        """Observed change outside the move's mask that forms piece-sized blobs (full-res px).

        Speckle, thin lines (a vinyl board's crease shifting) and exposure noise are opened
        away; what survives looks like a second moved or bumped piece.
        """
        residual = (change > 0.5) & ~mask
        opened = cv2.morphologyEx(residual.astype(np.uint8), cv2.MORPH_OPEN, self._blob_kernel)
        return float(change[opened > 0].sum()) * self.px

    def render(self, board: chess.Board) -> np.ndarray:
        return self.cam.render_labels(board, self.scale, self.roi)

    def predicted_mask(self, before_labels: np.ndarray, board: chess.Board, move: chess.Move) -> np.ndarray:
        after = board.copy(stack=False)
        after.push(move)
        return self.render(after) != before_labels

    def score_move(self, board: chess.Board, move: chess.Move, change: np.ndarray) -> tuple[Candidate, float]:
        """One move's score against a change map, and the unexplained share outside its mask."""
        change = np.where(self.region, change, 0.0).astype(np.float32)
        llr = change * self.reward + (1.0 - change) * self.penalty
        mask = self.predicted_mask(self.render(board), board, move)
        cand = Candidate(move, float(llr[mask].sum()) * self.px, float(mask.sum()) * self.px,
                         float(change[mask].sum()) * self.px)
        total = float(change.sum()) * self.px
        return cand, (self.unexplained_blobs(change, mask) / total if total > 0 else 0.0)

    def match(self, board: chess.Board, change: np.ndarray) -> MatchResult:
        """change: change probability map over the ROI at the work scale."""
        if change.shape != self.region.shape:
            raise ValueError(f"change map {change.shape} doesn't match region {self.region.shape}")
        change = np.where(self.region, change, 0.0).astype(np.float32)
        llr = change * self.reward + (1.0 - change) * self.penalty
        total = float(change.sum()) * self.px

        before = self.render(board)
        candidates: list[Candidate] = []
        promotion_seen = False
        masks: dict[chess.Move, np.ndarray] = {}
        for move in board.legal_moves:
            if move.promotion and move.promotion != chess.QUEEN:
                promotion_seen = True
                continue  # under-promotions look the same to the camera; decided separately
            mask = self.predicted_mask(before, board, move)
            masks[move] = mask
            candidates.append(Candidate(
                move=move,
                score=float(llr[mask].sum()) * self.px,
                mask_px=float(mask.sum()) * self.px,
                explained=float(change[mask].sum()) * self.px,
            ))
        candidates.sort(key=lambda c: c.score, reverse=True)
        result = MatchResult(candidates, total)
        if candidates:
            best = candidates[0]
            result.margin = best.score - (candidates[1].score if len(candidates) > 1 else 0.0)
            result.unexplained = self.unexplained_blobs(change, masks[best.move]) / total if total > 0 else 0.0
            result.is_promotion = best.move.promotion is not None
            if result.is_promotion and promotion_seen:
                result.notes.append("promotion: piece type decided by detector, default queen")
        return result
