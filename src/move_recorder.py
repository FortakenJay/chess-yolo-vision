"""Detect moves between stable positions and build game state."""

from __future__ import annotations

from collections.abc import Callable

import chess

from .errors import IllegalMoveError
from .types import GameState, MoveEvent, Orientation


class MoveRecorder:
    """Tracks legal moves by comparing incoming stable FEN positions."""

    def __init__(
        self,
        starting_fen: str = chess.STARTING_FEN,
        orientation: Orientation = Orientation.OVERHEAD_WHITE_BOTTOM,
        promotion_selector: Callable[[str], str] | None = None,
    ) -> None:
        self.orientation = orientation
        self.promotion_selector = promotion_selector
        self.board = chess.Board(starting_fen)
        self.starting_fen = starting_fen
        self.current_fen = self.board.fen()
        self.last_move: chess.Move | None = None
        self.moves: list[chess.Move] = []
        self.sans: list[str] = []

    def _promotion_piece(self, from_sq: str, to_sq: str) -> chess.PieceType | None:
        rank = chess.square_rank(chess.parse_square(to_sq))
        piece = self.board.piece_at(chess.parse_square(from_sq))
        if not piece or piece.piece_type != chess.PAWN:
            return None
        if rank not in {0, 7}:
            return None
        choice = "q"
        if self.promotion_selector:
            choice = self.promotion_selector(f"{from_sq}{to_sq}").strip().lower() or "q"
        mapping = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
        return mapping.get(choice, chess.QUEEN)

    def _candidate_moves(self, from_sq: str, to_sq: str) -> list[chess.Move]:
        moves = [chess.Move.from_uci(f"{from_sq}{to_sq}")]
        promotion = self._promotion_piece(from_sq, to_sq)
        if promotion is not None:
            moves = [chess.Move(chess.parse_square(from_sq), chess.parse_square(to_sq), promotion=promotion)]
        return moves

    def _find_legal_match(self, new_fen: str) -> chess.Move | None:
        target = chess.Board(new_fen).board_fen()
        for move in self.board.legal_moves:
            tmp = self.board.copy(stack=False)
            tmp.push(move)
            if tmp.board_fen() == target:
                return move
        return None

    def try_apply_fen(self, new_fen: str) -> MoveEvent | None:
        """Apply a new stable FEN if it is exactly one legal move ahead."""
        if not new_fen or not new_fen.strip():
            raise ValueError("new_fen must be non-empty")
        if self.board.is_game_over():
            return None
        if chess.Board(new_fen).board_fen() == self.board.board_fen():
            return None

        matched_move = self._find_legal_match(new_fen)
        if matched_move is None:
            raise IllegalMoveError("?", "?", new_fen)

        san = self.board.san(matched_move)
        self.board.push(matched_move)
        self.current_fen = self.board.fen()
        self.last_move = matched_move
        self.moves.append(matched_move)
        self.sans.append(san)

        side_to_move = "White" if self.board.turn == chess.WHITE else "Black"
        return MoveEvent(
            move_number=self.board.fullmove_number,
            side_to_move=side_to_move,
            move_uci=matched_move.uci(),
            move_san=san,
            from_square=chess.square_name(matched_move.from_square),
            to_square=chess.square_name(matched_move.to_square),
            fen_after=self.current_fen,
            move=matched_move,
        )

    def game_state(self) -> GameState:
        return GameState(
            played_at=None,  # set by pipeline/main at game start
            starting_fen=self.starting_fen,
            moves=self.moves.copy(),
            sans=self.sans.copy(),
        )

