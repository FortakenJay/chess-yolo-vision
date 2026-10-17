"""Legal position validation helpers."""

from __future__ import annotations

import logging

import chess

logger = logging.getLogger(__name__)


def is_legal_position(fen: str) -> bool:
    """Validate a FEN as a legal game position."""
    if not fen or not fen.strip():
        logger.debug("Rejected empty FEN")
        return False

    try:
        board = chess.Board(fen)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Rejected invalid FEN parse: %s", exc)
        return False

    pieces = board.board_fen()
    if "K" not in pieces or "k" not in pieces:
        logger.debug("Rejected: missing one or both kings")
        return False

    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq)
        if piece.piece_type == chess.PAWN and rank in {0, 7}:
            logger.debug("Rejected: pawn found on rank %s", rank + 1)
            return False

    if not board.is_valid():
        logger.debug("Rejected: board.is_valid() false")
        return False

    return True

