from datetime import datetime, timezone

import pytest

from src.errors import SupabaseInsertError
from src.supabase_client import SupabaseGameStore


class _FakeExec:
    def __init__(self, data):
        self.data = data


class _FakeTableSuccess:
    def insert(self, payload):
        return self

    def execute(self):
        return _FakeExec([{"id": "abc123"}])


class _FakeClientSuccess:
    def table(self, name):
        return _FakeTableSuccess()


class _FakeTableFail:
    def insert(self, payload):
        return self

    def execute(self):
        raise RuntimeError("boom")


class _FakeClientFail:
    def table(self, name):
        return _FakeTableFail()


def _payload():
    return {
        "pgn": "1. e4 *",
        "result": "white",
        "played_at": datetime.now(timezone.utc).isoformat(),
        "total_moves": 1,
    }


def test_save_success(monkeypatch, tmp_path):
    monkeypatch.setattr("src.supabase_client.create_client", lambda *_: _FakeClientSuccess())
    store = SupabaseGameStore("https://example.supabase.co", "key", failures_dir=str(tmp_path))
    inserted = store.save_game(_payload())
    assert inserted == "abc123"


def test_failed_insert_archives(monkeypatch, tmp_path):
    monkeypatch.setattr("src.supabase_client.create_client", lambda *_: _FakeClientFail())
    store = SupabaseGameStore("https://example.supabase.co", "key", failures_dir=str(tmp_path))
    with pytest.raises(SupabaseInsertError):
        store.save_game(_payload())
    assert list(tmp_path.glob("*.json"))

