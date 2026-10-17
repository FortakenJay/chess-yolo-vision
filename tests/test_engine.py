"""Finding Stockfish when config.yaml was written on another computer."""

from chessreader import engine


def test_configured_path_wins(tmp_path):
    exe = tmp_path / "stockfish"
    exe.write_text("")
    assert engine.find_stockfish(exe) == str(exe)


def test_windows_path_on_another_computer_falls_back_to_path(tmp_path, monkeypatch):
    exe = tmp_path / "stockfish"
    exe.write_text("")
    monkeypatch.setattr(engine.shutil, "which", lambda name: str(exe) if name == "stockfish" else None)
    assert engine.find_stockfish(r"C:\Users\Jay\stockfish-windows-x86-64-avx2.exe") == str(exe)


def test_none_found(tmp_path, monkeypatch):
    monkeypatch.setattr(engine.shutil, "which", lambda name: None)
    monkeypatch.setattr(engine, "_COMMON_PATHS", (str(tmp_path / "missing"),))
    monkeypatch.setattr("chessreader.settings.PROJECT_DIR", tmp_path)
    assert engine.find_stockfish(r"C:\nowhere\stockfish.exe") is None
