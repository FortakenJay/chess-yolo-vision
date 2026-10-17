import sys
import types
from pathlib import Path

import numpy as np

from src.pieces import PieceDetector


class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.array(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __getitem__(self, idx):
        if self._arr.ndim == 1:
            return _FakeScalar(float(self._arr[idx]))
        return self._arr[idx]


class _FakeBoxes:
    def __init__(self):
        self.xyxy = _FakeTensor([[100, 100, 160, 160], [200, 200, 260, 260]])
        self.conf = _FakeTensor([0.95, 0.96])
        self.cls = _FakeTensor([8, 2])  # white-king, black-king


class _FakeResult:
    def __init__(self):
        self.boxes = _FakeBoxes()
        self.names = {
            0: "bishop",
            1: "black-bishop",
            2: "black-king",
            3: "black-knight",
            4: "black-pawn",
            5: "black-queen",
            6: "black-rook",
            7: "white-bishop",
            8: "white-king",
            9: "white-knight",
            10: "white-pawn",
            11: "white-queen",
            12: "white-rook",
        }


class _FakeYOLO:
    def __init__(self, *_args, **_kwargs):
        pass

    def predict(self, **_kwargs):
        return [_FakeResult()]


def test_piece_detect_and_map(tmp_path: Path, monkeypatch):
    model = tmp_path / "pieces.pt"
    model.write_text("x", encoding="utf-8")
    fake_ultra = types.SimpleNamespace(YOLO=_FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)
    detector = PieceDetector(str(model), conf=0.4, warp_size=800)
    frame = np.zeros((800, 800, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    board_map = detector.to_board_map(detections)
    assert len(detections) == 2
    assert board_map is not None
    assert "K" in board_map.values()
    assert "k" in board_map.values()

