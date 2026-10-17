"""FEN building from board maps."""

from __future__ import annotations

from .errors import FENBuildError
from .types import BoardMap

VALID_PIECES = set("KQRBNPkqrbnp")


def build_fen(
    board_map: BoardMap,
    side_to_move: str = "w",
    castling: str = "KQkq",
    en_passant: str = "-",
    halfmove: int = 0,
    fullmove: int = 1,
) -> str:
    """Build a complete FEN string from board coordinates."""
    for key, value in board_map.items():
        if not (isinstance(key, tuple) and len(key) == 2 and all(isinstance(v, int) for v in key)):
            raise FENBuildError("board_map", f"invalid key type: {key}")
        r, c = key
        if not (0 <= r <= 7 and 0 <= c <= 7):
            raise FENBuildError(r, f"coordinate out of range: {key}")
        if value not in VALID_PIECES:
            raise FENBuildError(r, f"invalid piece char: {value}")

    ranks: list[str] = []
    for rank in range(8):
        empties = 0
        rank_parts: list[str] = []
        for file in range(8):
            piece = board_map.get((rank, file))
            if piece:
                if empties:
                    rank_parts.append(str(empties))
                    empties = 0
                rank_parts.append(piece)
            else:
                empties += 1
        if empties:
            rank_parts.append(str(empties))
        rank_str = "".join(rank_parts)
        squares_count = sum(int(ch) if ch.isdigit() else 1 for ch in rank_str)
        if squares_count != 8:
            raise FENBuildError(rank, rank_str)
        ranks.append(rank_str)

    if len(ranks) != 8:
        raise FENBuildError("board", "must have exactly 8 ranks")
    board_state = "/".join(ranks)
    if board_state.count("/") != 7:
        raise FENBuildError("board", "must have exactly 7 separators")

    castling_value = castling if castling else "-"
    return f"{board_state} {side_to_move} {castling_value} {en_passant} {halfmove} {fullmove}"

