"""Centralized rendering for camera and warped board windows."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from .types import DetectedPiece


def _draw_text_line(
    image: np.ndarray,
    text: str,
    line_index: int,
    color: tuple[int, int, int],
    font_scale: float,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 12, 24 + line_index * int(28 * font_scale + 8)
    (w, h), _ = cv2.getTextSize(text, font, font_scale, 1)
    cv2.rectangle(image, (x - 4, y - h - 6), (x + w + 4, y + 6), (20, 20, 20), -1)
    cv2.putText(image, text, (x, y), font, font_scale, color, 1, cv2.LINE_AA)


class DisplayRenderer:
    """Draw both windows from current pipeline state."""

    def __init__(
        self,
        warp_size: int,
        show_square_labels: bool = True,
        show_confidence_scores: bool = True,
        show_piece_count_hud: bool = True,
        highlight_last_move: bool = True,
        hud_font_scale: float = 0.6,
        grid_opacity: float = 0.5,
        highlight_opacity: float = 0.35,
    ) -> None:
        self.warp_size = warp_size
        self.show_square_labels = show_square_labels
        self.show_confidence_scores = show_confidence_scores
        self.show_piece_count_hud = show_piece_count_hud
        self.highlight_last_move = highlight_last_move
        self.hud_font_scale = hud_font_scale
        self.grid_opacity = grid_opacity
        self.highlight_opacity = highlight_opacity
        # BGR colors tuned for vivid high-contrast overlays.
        self.neon_green = (57, 255, 20)

    def _piece_color(self, label: str) -> tuple[int, int, int]:
        piece_name = label.split("-")[1] if "-" in label else label
        palette: dict[str, tuple[int, int, int]] = {
            "king": (0, 215, 255),    # gold
            "queen": (255, 0, 255),   # magenta
            "rook": (255, 128, 0),    # orange
            "bishop": (0, 255, 255),  # yellow
            "knight": (255, 0, 128),  # pink-red
            "pawn": (255, 255, 0),    # cyan
        }
        return palette.get(piece_name, (255, 255, 255))

    def _draw_neon_corner_dots(self, image: np.ndarray, corners: np.ndarray) -> None:
        for x, y in corners.astype(int):
            p = (int(x), int(y))
            # Outer glow + bright center for a "neon" look.
            cv2.circle(image, p, 14, self.neon_green, 2)
            cv2.circle(image, p, 6, self.neon_green, -1)

    def _draw_board_boundary(self, image: np.ndarray, corners: np.ndarray) -> None:
        pts = corners.astype(int).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], isClosed=True, color=self.neon_green, thickness=2)

    @staticmethod
    def _project_square_to_raw(
        file_idx: int,
        rank_idx: int,
        cell: int,
        inverse_h: np.ndarray,
    ) -> np.ndarray:
        x1 = file_idx * cell
        y1 = rank_idx * cell
        x2 = x1 + cell
        y2 = y1 + cell
        square = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
        return cv2.perspectiveTransform(square, inverse_h).astype(int)

    def render_raw(
        self,
        frame: np.ndarray,
        corners: np.ndarray | None,
        last_corners: np.ndarray | None,
        connection_status: str,
        move_history: str,
        move_number: int,
        side_to_move: str,
        game_status: str,
        last_move_san: str | None,
    ) -> np.ndarray:
        out = frame.copy()
        if corners is not None:
            self._draw_board_boundary(out, corners)
            self._draw_neon_corner_dots(out, corners)
        elif last_corners is not None:
            pts = last_corners.astype(int).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(out, "CORNERS LOST", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(out, "NO BOARD CORNERS - overlays paused", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        status_color = {
            "CONNECTED": (0, 255, 0),
            "RECONNECTING": (0, 255, 255),
            "LOST": (0, 0, 255),
        }.get(connection_status, (255, 255, 255))
        _draw_text_line(out, f"STREAM: {connection_status}", 0, status_color, self.hud_font_scale)
        _draw_text_line(out, f"PGN: {move_history}", 1, (0, 255, 255), self.hud_font_scale)
        _draw_text_line(out, f"Move {move_number} - {side_to_move} to move", 2, (255, 255, 255), self.hud_font_scale)
        _draw_text_line(out, f"Game: {game_status}", 3, (255, 255, 0), self.hud_font_scale)
        if last_move_san:
            _draw_text_line(out, f"Last move: {last_move_san}", 4, (255, 255, 255), self.hud_font_scale)
        return out

    def render_chess_vision(
        self,
        frame: np.ndarray,
        corners: np.ndarray | None,
        detections: list[DetectedPiece],
        inverse_h: np.ndarray | None,
        connection_status: str,
        move_history: str,
        move_number: int,
        side_to_move: str,
        game_status: str,
        last_move_san: str | None,
        last_corners: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render single-window chess vision overlay on the camera frame.

        Piece bounding boxes are projected from warped board coordinates back
        into the raw camera image using the inverse perspective matrix.
        Drawing order (bottom → top): raw image, board boundary, piece boxes.
        """
        out = self.render_raw(
            frame=frame,
            corners=corners,
            last_corners=last_corners,
            connection_status=connection_status,
            move_history=move_history,
            move_number=move_number,
            side_to_move=side_to_move,
            game_status=game_status,
            last_move_san=last_move_san,
        )

        _draw_text_line(
            out,
            f"detections: {len(detections)}",
            5,
            (200, 200, 200),
            self.hud_font_scale,
        )

        if not detections or inverse_h is None:
            return out

        for det in detections:
            color = self._piece_color(det.label)
            box_pts = np.array(
                [[[det.x1, det.y1], [det.x2, det.y1], [det.x2, det.y2], [det.x1, det.y2]]],
                dtype=np.float32,
            )
            projected = cv2.perspectiveTransform(box_pts, inverse_h).astype(int)
            cv2.polylines(out, [projected], isClosed=True, color=color, thickness=2)

            anchor_x, anchor_y = projected[0][0]
            tag = self._short_piece(det.label)
            if self.show_confidence_scores:
                tag = f"{tag} {det.confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            label_x = int(anchor_x)
            label_y = max(text_h + 4, int(anchor_y) - 4)
            cv2.rectangle(
                out,
                (label_x - 1, label_y - text_h - 4),
                (label_x + text_w + 4, label_y + 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                out,
                tag,
                (label_x + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        return out

    def _square_rect(self, square: str) -> tuple[int, int, int, int]:
        file_idx = ord(square[0]) - ord("a")
        rank = int(square[1])
        cell = self.warp_size // 8
        x1 = file_idx * cell
        y1 = (8 - rank) * cell
        return x1, y1, x1 + cell, y1 + cell

    def _draw_grid(self, board_img: np.ndarray) -> np.ndarray:
        overlay = board_img.copy()
        cell = self.warp_size // 8
        for i in range(9):
            p = i * cell
            cv2.line(overlay, (p, 0), (p, self.warp_size), (255, 255, 255), 1)
            cv2.line(overlay, (0, p), (self.warp_size, p), (255, 255, 255), 1)
        return cv2.addWeighted(overlay, self.grid_opacity, board_img, 1 - self.grid_opacity, 0)

    def _draw_square_labels(self, board_img: np.ndarray) -> None:
        if not self.show_square_labels:
            return
        cell = self.warp_size // 8
        for r in range(8):
            for c in range(8):
                sq = f"{chr(ord('a') + c)}{8 - r}"
                x = c * cell + 3
                y = (r + 1) * cell - 4
                cv2.rectangle(board_img, (x - 2, y - 12), (x + 22, y + 2), (0, 0, 0), -1)
                cv2.putText(board_img, sq, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    def _short_piece(self, label: str) -> str:
        side = "W" if label.startswith("white-") else "B"
        piece = label.split("-")[1][0].upper()
        if piece == "K" and "knight" in label:
            piece = "N"
        return f"{side}{piece}"

    def render_board(
        self,
        warped: np.ndarray,
        detections: list[DetectedPiece],
        last_move_from: str | None = None,
        last_move_to: str | None = None,
    ) -> np.ndarray:
        out = self._draw_grid(warped.copy())
        self._draw_square_labels(out)

        if self.highlight_last_move and last_move_from and last_move_to:
            overlay = out.copy()
            for square, color in ((last_move_from, (0, 180, 255)), (last_move_to, (0, 255, 255))):
                x1, y1, x2, y2 = self._square_rect(square)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            out = cv2.addWeighted(overlay, self.highlight_opacity, out, 1 - self.highlight_opacity, 0)

        for det in detections:
            is_white = det.label.startswith("white-")
            color = (255, 255, 255) if is_white else (50, 50, 50)
            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            if not is_white:
                cv2.rectangle(out, (det.x1 + 1, det.y1 + 1), (det.x2 - 1, det.y2 - 1), (255, 255, 255), 1)
            tag = self._short_piece(det.label)
            if self.show_confidence_scores:
                tag = f"{tag} {det.confidence:.2f}"
            (w, h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_bg = (245, 245, 245) if is_white else (40, 40, 40)
            tag_fg = (0, 0, 0) if is_white else (255, 255, 255)
            cv2.rectangle(out, (det.x1, max(0, det.y1 - h - 8)), (det.x1 + w + 6, det.y1), tag_bg, -1)
            cv2.putText(out, tag, (det.x1 + 3, det.y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, tag_fg, 1, cv2.LINE_AA)

        if self.show_piece_count_hud:
            counts = Counter([d.label for d in detections])
            strip_h = 46
            overlay = out.copy()
            cv2.rectangle(overlay, (0, self.warp_size - strip_h), (self.warp_size, self.warp_size), (0, 0, 0), -1)
            out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)
            white_count = sum(v for k, v in counts.items() if k.startswith("white-"))
            black_count = sum(v for k, v in counts.items() if k.startswith("black-"))
            cv2.putText(out, f"White pieces: {white_count}", (8, self.warp_size - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(out, f"Black pieces: {black_count}", (8, self.warp_size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return out

