"""OpenCV overlays for the debug windows."""

from __future__ import annotations

import cv2
import numpy as np

FILES = "abcdefgh"
GREEN = (0, 220, 0)
YELLOW = (0, 220, 255)
RED = (0, 0, 255)
CORNER_LABELS = ("TL", "TR", "BR", "BL")


def put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.6,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    """Text with a dark outline so it reads on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_hud(
    img: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 24), color: tuple[int, int, int] = (255, 255, 255)
) -> None:
    x, y = origin
    for line in lines:
        put_text(img, line, (x, y), color=color)
        y += 24


def draw_quad(img: np.ndarray, corners: np.ndarray, color: tuple[int, int, int] = GREEN) -> None:
    """Outline the board; corners are in warp order TL, TR, BR, BL."""
    pts = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    for label, (x, y) in zip(CORNER_LABELS, np.round(corners).astype(int), strict=True):
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


def draw_mask_outline(
    frame: np.ndarray, mask: np.ndarray, roi: tuple[int, int, int, int], color: tuple[int, int, int] = YELLOW
) -> None:
    x0, y0, x1, y1 = roi
    big = cv2.resize(mask.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shifted = [c + np.array([[[x0, y0]]], dtype=c.dtype) for c in contours]
    cv2.drawContours(frame, shifted, -1, color, 2, cv2.LINE_AA)


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
    bar[h - white_h :] = 235
    cv2.line(bar, (0, h // 2), (width, h // 2), (0, 0, 255), 1)
    return np.hstack([bar, img])


def _blend(img: np.ndarray, overlay: np.ndarray, alpha: float) -> None:
    """img <- overlay * alpha + img * (1 - alpha), in place (img may be a slice)."""
    img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def square_rect(square: int, size: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of a square in the white-at-bottom top-down view."""
    import chess

    sq = size / 8
    col, row = chess.square_file(square), 7 - chess.square_rank(square)
    return int(round(col * sq)), int(round(row * sq)), int(round((col + 1) * sq)), int(round((row + 1) * sq))


def highlight_squares(
    warp: np.ndarray, squares, color: tuple[int, int, int] = (0, 200, 230), alpha: float = 0.35
) -> None:
    """Semi-transparent fill of whole squares, like chess sites mark the last move."""
    for square in squares:
        x0, y0, x1, y1 = square_rect(square, warp.shape[0])
        region = warp[y0:y1, x0:x1]
        tint = np.empty_like(region)
        tint[:] = color
        _blend(region, tint, alpha)


def highlight_move(warp: np.ndarray, move, color: tuple[int, int, int] = (0, 200, 230), alpha: float = 0.35) -> None:
    if move is not None:
        highlight_squares(warp, (move.from_square, move.to_square), color, alpha)


def draw_soft_arrow(
    warp: np.ndarray, move, color: tuple[int, int, int] = YELLOW, alpha: float = 0.5, width: float = 0.07
) -> None:
    """Thin semi-transparent arrow from square center to square center; width is a fraction of a square."""
    if move is None:
        return
    size = warp.shape[0]
    sq = size / 8
    a = np.array(square_px(move.from_square, size), np.float64)
    b = np.array(square_px(move.to_square, size), np.float64)
    d = b - a
    length = float(np.hypot(*d))
    if length < 1:
        return
    u = d / length
    n = np.array([-u[1], u[0]])
    head_len, head_w = sq * 0.32, sq * 0.24
    base = b - u * min(head_len, length * 0.6)
    overlay = warp.copy()
    shaft = max(2, int(round(sq * width)))
    cv2.line(overlay, tuple(np.round(a).astype(int)), tuple(np.round(base).astype(int)), color, shaft, cv2.LINE_AA)
    head = np.array([b, base + n * head_w, base - n * head_w])
    cv2.fillConvexPoly(overlay, np.round(head).astype(np.int32), color, cv2.LINE_AA)
    _blend(warp, overlay, alpha)


BADGE_CORNERS = {"tr": (1, -1), "br": (1, 1), "bl": (-1, 1), "tl": (-1, -1)}  # image x, y directions


def badge_corner(move, taken=()) -> str:
    """Corner of the destination square facing away from the arrow's tail, so the arrow doesn't run over it.

    Top-left is a last resort (the grid's square labels sit there); corners in taken are skipped.
    """
    import chess

    dx = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    dy = chess.square_rank(move.from_square) - chess.square_rank(move.to_square)  # image y grows downwards
    free = [c for c in BADGE_CORNERS if c not in taken] or list(BADGE_CORNERS)
    return max(free, key=lambda c: BADGE_CORNERS[c][0] * dx + BADGE_CORNERS[c][1] * dy - (2.5 if c == "tl" else 0))


def draw_badge(
    warp: np.ndarray,
    square: int,
    text: str,
    color: tuple[int, int, int] = YELLOW,
    corner: str = "tr",
    scale: float = 1.0,
) -> None:
    """Small filled circle with a label, tucked into a corner of the square so the piece stays visible."""
    x0, y0, x1, y1 = square_rect(square, warp.shape[0])
    sq = x1 - x0
    r = max(7, int(round(sq * 0.15 * scale)))
    pad = int(round(sq * 0.05)) + r
    cx = x1 - pad if corner[1] == "r" else x0 + pad
    cy = y0 + pad if corner[0] == "t" else y1 - pad
    cv2.circle(warp, (cx, cy), r + 2, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(warp, (cx, cy), r, color, -1, cv2.LINE_AA)
    font_scale = r / 22
    thickness = max(1, int(round(r / 9)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(
        warp,
        text,
        (cx - tw // 2, cy + th // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )


def draw_choices(warp: np.ndarray, moves, color: tuple[int, int, int] = YELLOW) -> None:
    """Candidate moves for a question: faint arrows, numbered badges on the destinations (1 a bit stronger)."""
    for i, move in reversed(list(enumerate(moves))):  # the first choice ends up on top
        draw_soft_arrow(warp, move, color, alpha=0.7 if i == 0 else 0.45, width=0.08 if i == 0 else 0.06)
    taken: dict[int, list[str]] = {}
    for i, move in enumerate(moves):
        corner = badge_corner(move, taken.setdefault(move.to_square, []))
        taken[move.to_square].append(corner)
        draw_badge(warp, move.to_square, str(i + 1), color, corner, scale=1.0 if i == 0 else 0.9)
