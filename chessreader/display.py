"""OpenCV overlays for the debug windows."""

from __future__ import annotations

import cv2
import numpy as np

FILES = "abcdefgh"
GREEN = (0, 220, 0)
YELLOW = (0, 220, 255)
RED = (0, 0, 255)
CORNER_LABELS = ("TL", "TR", "BR", "BL")


def put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.6,
             color: tuple[int, int, int] = (255, 255, 255), thickness: int = 1) -> None:
    """Text with a dark outline so it reads on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_hud(img: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 24),
             color: tuple[int, int, int] = (255, 255, 255)) -> None:
    x, y = origin
    for line in lines:
        put_text(img, line, (x, y), color=color)
        y += 24


def draw_quad(img: np.ndarray, corners: np.ndarray, color: tuple[int, int, int] = GREEN) -> None:
    """Outline the board; corners are in warp order TL, TR, BR, BL."""
    pts = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    for label, (x, y) in zip(CORNER_LABELS, np.round(corners).astype(int)):
        cv2.circle(img, (x, y), 5, color, -1, cv2.LINE_AA)
        put_text(img, label, (x + 6, y - 6), scale=0.5, color=color)


def draw_points(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int] = YELLOW, radius: int = 3) -> None:
    for x, y in np.round(pts.reshape(-1, 2)).astype(int):
        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)


def draw_grid(warp: np.ndarray, show_labels: bool = True, color: tuple[int, int, int] = GREEN) -> None:
    """8x8 grid on the top-down board. Labels assume white sits at the bottom."""
    size = warp.shape[0]
    sq = size / 8
    for i in range(9):
        p = int(round(i * sq))
        cv2.line(warp, (p, 0), (p, size), color, 1, cv2.LINE_AA)
        cv2.line(warp, (0, p), (size, p), color, 1, cv2.LINE_AA)
    if show_labels:
        for row in range(8):
            for col in range(8):
                name = f"{FILES[col]}{8 - row}"
                put_text(warp, name, (int(col * sq) + 4, int(row * sq) + 16), scale=0.4, color=color)


def square_px(square: int, size: int) -> tuple[int, int]:
    """Center of a square in the white-at-bottom top-down view."""
    import chess

    sq = size / 8
    return int((chess.square_file(square) + 0.5) * sq), int((7 - chess.square_rank(square) + 0.5) * sq)


def draw_change(frame: np.ndarray, change: np.ndarray, roi: tuple[int, int, int, int], alpha: float = 0.45) -> None:
    """Tint pixels that changed since the last accepted position."""
    x0, y0, x1, y1 = roi
    big = cv2.resize(change, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    region = frame[y0:y1, x0:x1]
    tint = np.zeros_like(region)
    tint[..., 2] = 255
    weight = (big * alpha)[..., None]
    frame[y0:y1, x0:x1] = (region * (1 - weight) + tint * weight).astype(np.uint8)


def draw_mask_outline(frame: np.ndarray, mask: np.ndarray, roi: tuple[int, int, int, int],
                      color: tuple[int, int, int] = YELLOW) -> None:
    x0, y0, x1, y1 = roi
    big = cv2.resize(mask.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        c += np.array([[[x0, y0]]], dtype=c.dtype)
    cv2.drawContours(frame, contours, -1, color, 2, cv2.LINE_AA)


def draw_occupancy(warp: np.ndarray, probs: np.ndarray | None) -> None:
    """Dot per square: white/black = what the camera sees, size = confidence, hidden squares skipped."""
    if probs is None:
        return
    sq = warp.shape[0] / 8
    for r in range(8):
        for c in range(8):
            p = probs[r, c]
            if np.isnan(p).any():
                continue
            k = int(np.argmax(p))
            if k == 0 or p[k] < 0.6:  # empty, or too unsure to show
                continue
            center = (int((c + 0.5) * sq), int((r + 0.5) * sq))
            fill = (240, 240, 240) if k == 1 else (20, 20, 20)
            cv2.circle(warp, center, int(sq * 0.12 * (0.5 + p[k])), fill, -1, cv2.LINE_AA)
            cv2.circle(warp, center, int(sq * 0.12 * (0.5 + p[k])), (0, 160, 255), 1, cv2.LINE_AA)


def draw_move(warp: np.ndarray, move, color: tuple[int, int, int] = (0, 200, 255)) -> None:
    if move is None:
        return
    size = warp.shape[0]
    a, b = square_px(move.from_square, size), square_px(move.to_square, size)
    cv2.arrowedLine(warp, a, b, color, 4, cv2.LINE_AA, tipLength=0.25)


def draw_eval_bar(img: np.ndarray, eval_cp: int | None, mate: int | None, width: int = 26) -> np.ndarray:
    """Return img with a white/black eval bar on its left."""
    h = img.shape[0]
    if mate is not None:
        frac = 1.0 if mate > 0 else 0.0
    elif eval_cp is not None:
        frac = 1 / (1 + 10 ** (-eval_cp / 400))
    else:
        frac = 0.5
    bar = np.full((h, width, 3), 30, np.uint8)
    white_h = int(h * frac)
    bar[h - white_h:] = 235
    cv2.line(bar, (0, h // 2), (width, h // 2), (0, 0, 255), 1)
    return np.hstack([bar, img])
