"""Legal-move matching: which legal move best explains what changed on the board?

For each legal move, the camera model renders which piece is visible at every pixel before
and after the move (nearer pieces hide farther ones). Pixels whose visible piece differs
form the move's predicted change mask M. With b(p) the observed change probability:

    score(move) = sum over p in M of  b(p) * log(q_in / q_out) + (1 - b(p)) * log((1 - q_in) / (1 - q_out))

which is the log-likelihood ratio of "this move happened" against "nothing happened"
(q_in: chance a pixel inside the mask visibly changes, q_out: noise rate outside it).
q_in depends on what a pixel shows before and after: a black piece over a dark square (or
white over light) changes a pixel far less often than black over light. How faint those pairs
are depends on the board and the light, so it is measured at game start (set_tone_contrast).
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

from .geometry import BoardCamera, square_center

logger = logging.getLogger(__name__)

# what a pixel shows: off the board, a light or dark square, a white or black piece
TABLE, LIGHT, DARK, WHITE, BLACK = range(5)


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
    off_board: float = 0.0       # change beyond the board edge in hand-sized blobs (full-res px)
    is_promotion: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def top(self, n: int = 3) -> list[Candidate]:
        return self.candidates[:n]


class MoveMatcher:
    def __init__(self, cam: BoardCamera, roi: tuple[int, int, int, int], scale: float,
                 q_in: float = 0.55, q_out: float = 0.03, q_in_low: float = 0.25,
                 off_board_weight: float = 1.0) -> None:
        self.cam = cam
        self.roi = roi
        self.scale = scale
        self.q_in, self.q_out, self.q_in_low = q_in, q_out, q_in_low
        self.reward = math.log(q_in / q_out)
        self.penalty = math.log((1 - q_in) / (1 - q_out))
        self.px = 1.0 / (scale * scale)  # one work pixel in full-res pixels
        self.region = self._region_mask()
        self.floor = self._floor_tones()
        # beyond the board edge only the tops of tall pieces can change, and hands and sleeves rest
        # there: that evidence may support a move the board itself shows, never make one up
        self.off_board_weight = off_board_weight
        self.on_board = self.floor != TABLE
        self._piece_tone = np.array([TABLE] + [WHITE if i % 12 < 6 else BLACK for i in range(64 * 12)], np.uint8)
        q = np.full((5, 5), q_in)
        self._set_q(q)
        # a blob has to be at least ~1/3 of a square across to count as a piece-sized change
        a, b = cam.project(np.array([[4.0, 4.0, 0.0], [5.0, 4.0, 0.0]], np.float64))
        k = max(3, int(round(np.linalg.norm(b - a) * scale / 3)) | 1)
        self._blob_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def _set_q(self, q: np.ndarray) -> None:
        self._q = q
        self._pair_reward = np.log(q / self.q_out).ravel().astype(np.float32)
        self._pair_penalty = np.log((1 - q) / (1 - self.q_out)).ravel().astype(np.float32)

    def set_tone_contrast(self, q_black_on_dark: float, q_white_on_light: float) -> None:
        """Expected change rates for a piece over a square of its own tone, measured on this board."""
        q = self._q.copy()
        q[DARK, BLACK] = q[BLACK, DARK] = q_black_on_dark
        q[LIGHT, WHITE] = q[WHITE, LIGHT] = q_white_on_light
        self._set_q(q)

    def q_for_contrast(self, distance: float, t_low: float, t_high: float) -> float:
        """Faint pairs (colour distance near the change threshold) get q_in_low, clear ones q_in."""
        k = float(np.clip((distance - t_low) / (t_high - t_low), 0.0, 1.0))
        return self.q_in_low + k * (self.q_in - self.q_in_low)

    def _region_mask(self) -> np.ndarray:
        """Pixels any piece on any square could cover: the only place moves can change anything."""
        board = chess.Board(None)
        for sq in chess.SQUARES:
            board.set_piece_at(sq, chess.Piece(chess.KING, chess.WHITE))
        return self.render(board) > 0

    def _floor_tones(self) -> np.ndarray:
        """Tone of the board square under each pixel (TABLE off the board)."""
        x0, y0 = self.roi[0], self.roi[1]
        out = np.zeros(self.region.shape, np.uint8)
        for sq in chess.SQUARES:
            cx, cy = square_center(sq)
            corners = np.array([[cx - .5, cy - .5, 0], [cx + .5, cy - .5, 0], [cx + .5, cy + .5, 0], [cx - .5, cy + .5, 0]])
            poly = np.round((self.cam.project(corners) - [x0, y0]) * self.scale).astype(np.int32)
            light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
            cv2.fillPoly(out, [poly.reshape(-1, 1, 2)], LIGHT if light else DARK)
        return out

    def tones(self, labels: np.ndarray) -> np.ndarray:
        return np.where(labels > 0, self._piece_tone[np.maximum(labels, 0)], self.floor)

    def _candidate(self, move: chess.Move, mask: np.ndarray, before_tones: np.ndarray, after_labels: np.ndarray,
                   change: np.ndarray) -> Candidate:
        pair = before_tones[mask].astype(np.intp) * 5 + self.tones(after_labels)[mask]
        b = change[mask]
        llr = b * self._pair_reward[pair] + (1.0 - b) * self._pair_penalty[pair]
        on = self.on_board[mask]
        on_score = float(llr[on].sum())
        off_score = float(llr[~on].sum()) * self.off_board_weight
        score = (on_score + min(off_score, max(on_score, 0.0))) * self.px
        return Candidate(move, score, float(mask.sum()) * self.px, float(b.sum()) * self.px)

    def unexplained_blobs(self, change: np.ndarray, mask: np.ndarray) -> float:
        """Observed change outside the move's mask that forms piece-sized blobs (full-res px).

        Speckle, thin lines (a vinyl board's crease shifting) and exposure noise are opened
        away; what survives looks like a second moved or bumped piece.
        """
        residual = (change > 0.5) & ~mask & (self.floor != TABLE)
        opened = cv2.morphologyEx(residual.astype(np.uint8), cv2.MORPH_OPEN, self._blob_kernel)
        return float(change[opened > 0].sum()) * self.px

    def off_board_blobs(self, change: np.ndarray, mask: np.ndarray) -> float:
        """Change beyond the board edge, outside the move's own mask, that forms blobs: a hand or arm nearby."""
        residual = (change > 0.5) & (self.floor == TABLE) & ~mask
        opened = cv2.morphologyEx(residual.astype(np.uint8), cv2.MORPH_OPEN, self._blob_kernel)
        return float(change[opened > 0].sum()) * self.px

    def render(self, board: chess.Board) -> np.ndarray:
        return self.cam.render_labels(board, self.scale, self.roi)

    def predicted_mask(self, before_labels: np.ndarray, board: chess.Board, move: chess.Move) -> np.ndarray:
        return self._after(before_labels, board, move)[0]

    def _after(self, before_labels: np.ndarray, board: chess.Board, move: chess.Move) -> tuple[np.ndarray, np.ndarray]:
        """(predicted change mask, after-move labels)."""
        after = board.copy(stack=False)
        after.push(move)
        labels = self.render(after)
        return labels != before_labels, labels

    def score_move(self, board: chess.Board, move: chess.Move, change: np.ndarray) -> tuple[Candidate, float]:
        """One move's score against a change map, and the unexplained share outside its mask."""
        change = np.where(self.region, change, 0.0).astype(np.float32)
        before = self.render(board)
        mask, after = self._after(before, board, move)
        cand = self._candidate(move, mask, self.tones(before), after, change)
        total = float(change.sum()) * self.px  # all of it: a hand resting off the board still counts
        return cand, (self.unexplained_blobs(change, mask) / total if total > 0 else 0.0)

    def match(self, board: chess.Board, change: np.ndarray) -> MatchResult:
        """change: change probability map over the ROI at the work scale."""
        if change.shape != self.region.shape:
            raise ValueError(f"change map {change.shape} doesn't match region {self.region.shape}")
        change = np.where(self.region, change, 0.0).astype(np.float32)
        total = float(change.sum()) * self.px  # all of it: a hand resting off the board still counts

        before = self.render(board)
        before_tones = self.tones(before)
        candidates: list[Candidate] = []
        promotion_seen = False
        masks: dict[chess.Move, np.ndarray] = {}
        for move in board.legal_moves:
            if move.promotion and move.promotion != chess.QUEEN:
                promotion_seen = True
                continue  # under-promotions look the same to the camera; decided separately
            mask, after = self._after(before, board, move)
            masks[move] = mask
            candidates.append(self._candidate(move, mask, before_tones, after, change))
        candidates.sort(key=lambda c: c.score, reverse=True)
        result = MatchResult(candidates, total)
        if candidates:
            best = candidates[0]
            result.margin = best.score - (candidates[1].score if len(candidates) > 1 else 0.0)
            result.unexplained = self.unexplained_blobs(change, masks[best.move]) / total if total > 0 else 0.0
            result.off_board = self.off_board_blobs(change, masks[best.move])
            result.is_promotion = best.move.promotion is not None
            if result.is_promotion and promotion_seen:
                result.notes.append("promotion: piece type decided by detector, default queen")
        return result
