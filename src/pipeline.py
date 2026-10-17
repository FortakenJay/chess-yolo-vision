"""End-to-end chess vision runtime orchestration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from urllib.parse import quote

import chess
import cv2
import numpy as np

from .corners import CornerDetector
from .display import DisplayRenderer
from .fen import build_fen
from .move_recorder import MoveRecorder
from .pgn_exporter import export_game
from .pieces import PieceDetector
from .settings import AppSettings
from .smoother import FENSmoother
from .stream import MJPEGStreamReader
from .supabase_client import SupabaseGameStore
from .types import Orientation
from .validator import is_legal_position
from .warp import warp_board

logger = logging.getLogger(__name__)


class GameStatus(str, Enum):
    WAITING = "WAITING"
    IN_PROGRESS = "IN PROGRESS"
    ENDED = "ENDED"


class ChessVisionPipeline:
    """Coordinates stream ingestion, detection, validation, and recording."""

    def __init__(
        self,
        settings: AppSettings,
        orientation: Orientation = Orientation.OVERHEAD_WHITE_BOTTOM,
    ) -> None:
        self.settings = settings
        cfg = settings.runtime

        print(f"Connecting to stream {cfg.esp32_url} …")
        self.stream = MJPEGStreamReader(
            url=cfg.esp32_url,
            min_width=cfg.min_frame_width,
            min_height=cfg.min_frame_height,
        )
        print("Stream connected. Initializing ArUco corner detector …")
        self.corner_detector = CornerDetector(min_area_px=cfg.min_board_area_px)
        print("Loading piece model …")
        self.piece_detector = PieceDetector(cfg.piece_model_path, cfg.piece_conf, cfg.warp_size, orientation=orientation)
        print("Models loaded. Opening display windows …")
        self.smoother = FENSmoother(cfg.smoothing_frames)
        self.display = DisplayRenderer(
            warp_size=cfg.warp_size,
            show_square_labels=cfg.show_square_labels,
            show_confidence_scores=cfg.show_confidence_scores,
            show_piece_count_hud=cfg.show_piece_count_hud,
            highlight_last_move=cfg.highlight_last_move,
            hud_font_scale=cfg.hud_font_scale,
            grid_opacity=cfg.grid_opacity,
            highlight_opacity=cfg.highlight_opacity,
        )
        self.store = SupabaseGameStore(
            url=settings.env.SUPABASE_URL,
            key=settings.env.SUPABASE_KEY,
            failures_dir=cfg.failures_dir,
        )

        self.orientation = orientation
        self.last_corners = None
        self.last_stable_fen = chess.STARTING_FEN
        self.last_move_san: str | None = None
        self.last_move_from: str | None = None
        self.last_move_to: str | None = None
        self.played_at: datetime | None = None
        self.status = GameStatus.WAITING
        self.recorder = MoveRecorder(orientation=orientation)
        self._latest_detections = []
        self.connection_status = "CONNECTED"

    def start_game(self) -> None:
        self.recorder = MoveRecorder(orientation=self.orientation)
        self.played_at = datetime.now(timezone.utc)
        self.status = GameStatus.IN_PROGRESS
        self.last_stable_fen = chess.STARTING_FEN
        self.last_move_san = None
        self.last_move_from = None
        self.last_move_to = None
        print("Game started. Waiting for first move...")

    def abort_game(self) -> None:
        self.status = GameStatus.WAITING
        self.recorder = MoveRecorder(orientation=self.orientation)
        print("Game aborted.")

    def _result_to_pgn(self, result: str) -> str:
        return {"white": "1-0", "black": "0-1", "draw": "1/2-1/2", "unknown": "*"}.get(result, "*")

    def _format_pgn_for_hud(self, max_plies: int = 10) -> str:
        sans = self.recorder.sans
        if not sans:
            return "(no moves yet)"
        truncated = sans[-max_plies:]
        offset = len(sans) - len(truncated)
        parts: list[str] = []
        for i, san in enumerate(truncated):
            ply_idx = offset + i
            if ply_idx % 2 == 0:
                parts.append(f"{ply_idx // 2 + 1}.{san}")
            else:
                parts.append(san)
        prefix = "... " if offset > 0 else ""
        return prefix + " ".join(parts)

    def end_game(self, result: str) -> None:
        if self.status != GameStatus.IN_PROGRESS:
            return
        if not self.recorder.moves:
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"GAME OVER — {result.capitalize()} (no moves recorded; nothing to save)")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("Press 's' to start a new game.")
            self.status = GameStatus.ENDED
            return
        pgn_result = self._result_to_pgn(result)
        export = export_game(
            moves=self.recorder.moves,
            starting_fen=self.recorder.starting_fen,
            result=pgn_result,
            exports_dir=self.settings.runtime.exports_dir,
            opening_book_path=self.settings.runtime.opening_book_path,
        )
        payload = {
            "played_at": (self.played_at or datetime.now(timezone.utc)).isoformat(),
            "result": result if result in {"white", "black", "draw", "unknown"} else "unknown",
            "opening_name": export.opening_name,
            "opening_eco": export.opening_eco,
            "pgn": export.pgn,
            "starting_fen": self.recorder.starting_fen,
            "total_moves": len(self.recorder.moves),
            "capture_device": "esp32-cam",
            "notes": None,
        }
        supabase_ok = True
        try:
            self.store.save_game(payload)
        except Exception as exc:  # noqa: BLE001
            supabase_ok = False
            logger.error("Supabase save failed: %s", exc)

        lichess_url = f"https://lichess.org/analysis/pgn/{quote(export.pgn)}"
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"GAME OVER — {result.capitalize()}")
        print(f"Opening: {export.opening_name or 'Unknown'} ({export.opening_eco or 'N/A'})")
        print(f"Total moves: {len(self.recorder.moves)}")
        print(f"PGN saved: {export.file_path}")
        print(f"Supabase: {'✓ saved' if supabase_ok else 'fallback archive only'}")
        print(f"Lichess: {lichess_url}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("Game saved. Press 's' to start a new game.")
        self.status = GameStatus.ENDED

    def process_frame(self) -> np.ndarray | None:
        frame = self.stream.read_frame()
        if self.settings.runtime.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        corners = self.corner_detector.detect(frame)
        if corners is None:
            return self.display.render_chess_vision(
                frame=frame,
                corners=None,
                detections=[],
                inverse_h=None,
                last_corners=self.last_corners,
                connection_status=self.connection_status,
                move_history=self._format_pgn_for_hud(),
                move_number=self.recorder.board.fullmove_number,
                side_to_move="White" if self.recorder.board.turn else "Black",
                game_status=self.status.value,
                last_move_san=self.last_move_san,
            )

        self.last_corners = corners
        warped, h_matrix = warp_board(frame, corners, self.settings.runtime.warp_size)
        detections = self.piece_detector.detect(warped)
        if detections:
            top_conf = max(d.confidence for d in detections)
            logger.info("Detected %d pieces (top conf: %.2f)", len(detections), top_conf)
        if getattr(self, "_show_warped_debug", False):
            cv2.imshow("Warped (model input)", warped)
        board_map = self.piece_detector.to_board_map(detections)
        if board_map:
            fen = build_fen(
                board_map,
                side_to_move="w" if self.recorder.board.turn else "b",
                castling=self.recorder.board.castling_xfen(),
                en_passant=chess.square_name(self.recorder.board.ep_square) if self.recorder.board.ep_square else "-",
                halfmove=self.recorder.board.halfmove_clock,
                fullmove=self.recorder.board.fullmove_number,
            )
            if is_legal_position(fen):
                self.smoother.add(fen)
                stable = self.smoother.stable()
                if stable and stable != self.last_stable_fen and self.status == GameStatus.IN_PROGRESS:
                    event = self.recorder.try_apply_fen(stable)
                    if event:
                        self.last_stable_fen = stable
                        self.last_move_san = event.move_san
                        self.last_move_from = event.from_square
                        self.last_move_to = event.to_square
                        who = "White" if not self.recorder.board.turn else "Black"
                        print(f"[Move {len(self.recorder.moves)} | {who}] {event.move_san}  ->  {event.from_square}->{event.to_square}")
                        print(f"FEN: {event.fen_after}")
                        print("----------------------------------------------------")

        inverse_h = None
        try:
            inverse_h = np.linalg.inv(h_matrix)
        except Exception:  # noqa: BLE001
            inverse_h = None

        vision_view = self.display.render_chess_vision(
            frame=frame,
            corners=corners,
            detections=detections,
            inverse_h=inverse_h,
            last_corners=self.last_corners,
            connection_status=self.connection_status,
            move_history=self._format_pgn_for_hud(),
            move_number=self.recorder.board.fullmove_number,
            side_to_move="White" if self.recorder.board.turn else "Black",
            game_status=self.status.value,
            last_move_san=self.last_move_san,
        )
        self._latest_detections = detections
        return vision_view

    def run(self) -> None:
        cv2.namedWindow("Chess Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chess Vision", 1280, 720)
        self._show_warped_debug = False
        print(
            "Click the 'Chess Vision' window to focus it, then use keys: "
            "q=quit  s=start  a=abort  r=resign  e=end-game  w=toggle warped-debug window"
        )
        try:
            while True:
                view = self.process_frame()
                if view is not None:
                    cv2.imshow("Chess Vision", view)

                key = cv2.waitKey(1) & 0xFF
                if key == 0xFF:
                    continue
                if key == ord("q"):
                    print("Quitting…")
                    break
                if key == ord("s"):
                    self.start_game()
                elif key == ord("a"):
                    self.abort_game()
                elif key == ord("r"):
                    resigning = input("Who resigns? (w/b): ").strip().lower()
                    self.end_game("black" if resigning == "w" else "white")
                elif key == ord("e"):
                    result = input("Result? (w/b/d): ").strip().lower()
                    self.end_game({"w": "white", "b": "black", "d": "draw"}.get(result, "unknown"))
                elif key == ord("w"):
                    self._show_warped_debug = not self._show_warped_debug
                    if not self._show_warped_debug:
                        cv2.destroyWindow("Warped (model input)")
                    print(f"Warped debug window: {'ON' if self._show_warped_debug else 'OFF'}")
                else:
                    logger.debug("Unhandled key: %s", key)
        finally:
            self.stream.close()
            cv2.destroyAllWindows()

