"""Shared dataclasses and type aliases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

import chess

BoardMap = dict[tuple[int, int], str]
GameResult = Literal["white", "black", "draw", "unknown"]


class Orientation(str, Enum):
    """Camera/board orientation used to map pixel coordinates to chess squares.

    OVERHEAD_WHITE_BOTTOM – top-down camera, white pieces at the bottom of the frame.
    OVERHEAD_WHITE_TOP    – top-down camera, white pieces at the top of the frame.
    SIDE_WHITE_LEFT       – side/tripod camera, white pieces on the LEFT of the frame.
    SIDE_WHITE_RIGHT      – side/tripod camera, white pieces on the RIGHT of the frame.
    """

    OVERHEAD_WHITE_BOTTOM = "overhead_white_bottom"
    OVERHEAD_WHITE_TOP = "overhead_white_top"
    SIDE_WHITE_LEFT = "side_white_left"
    SIDE_WHITE_RIGHT = "side_white_right"


@dataclass(slots=True)
class DetectedPiece:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    square: str | None = None


@dataclass(slots=True)
class MoveEvent:
    move_number: int
    side_to_move: str
    move_uci: str
    move_san: str
    from_square: str
    to_square: str
    fen_after: str
    move: chess.Move


@dataclass(slots=True)
class GameState:
    played_at: datetime | None
    starting_fen: str
    moves: list[chess.Move]
    sans: list[str]


@dataclass(slots=True)
class ExportResult:
    pgn: str
    file_path: str
    opening_name: str | None
    opening_eco: str | None

