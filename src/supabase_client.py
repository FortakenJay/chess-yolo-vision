"""Supabase persistence with local fallback archive."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from supabase import Client, create_client

from .errors import SupabaseInsertError
from .types import GameResult

logger = logging.getLogger(__name__)

VALID_RESULTS: set[GameResult] = {"white", "black", "draw", "unknown"}


class SupabaseGameStore:
    """Insert game records into Supabase with fallback archive."""

    def __init__(self, url: str, key: str, table: str = "games", failures_dir: str = "exports/failures") -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Invalid SUPABASE_URL format")
        if not key:
            raise ValueError("SUPABASE_KEY is required")
        self.table = table
        self.client: Client = create_client(url, key)
        self.failures_dir = Path(failures_dir)
        self.failures_dir.mkdir(parents=True, exist_ok=True)

    def _validate_payload(self, payload: dict) -> None:
        pgn = payload.get("pgn")
        result = payload.get("result")
        played_at = payload.get("played_at")
        total_moves = payload.get("total_moves")

        if not isinstance(pgn, str) or not pgn.strip():
            raise ValueError("payload.pgn must be a non-empty string")
        if result not in VALID_RESULTS:
            raise ValueError("payload.result must be white|black|draw|unknown")
        try:
            datetime.fromisoformat(str(played_at).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("payload.played_at must be ISO8601") from exc
        if not isinstance(total_moves, int) or total_moves < 0:
            raise ValueError("payload.total_moves must be int >= 0")

    def _archive_failure(self, payload: dict, reason: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        path = self.failures_dir / f"{timestamp}.json"
        blob = {"reason": reason, "payload": payload}
        path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        return path

    def save_game(self, payload: dict) -> str:
        """Insert payload and return inserted id."""
        self._validate_payload(payload)
        try:
            response = self.client.table(self.table).insert(payload).execute()
            data = getattr(response, "data", None)
            if not data:
                raise RuntimeError("Supabase returned empty data")
            inserted = data[0]
            return str(inserted.get("id", ""))
        except Exception as exc:  # noqa: BLE001
            archived = self._archive_failure(payload, str(exc))
            logger.error("Supabase insert failed. Archived payload at %s. Error: %s", archived, exc)
            raise SupabaseInsertError(self.table, str(exc)) from exc

