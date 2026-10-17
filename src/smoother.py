"""Temporal FEN smoother."""

from __future__ import annotations

from collections import Counter, deque


class FENSmoother:
    """Keep a sliding window of FEN values and return the consensus."""

    def __init__(self, window_size: int) -> None:
        if not (2 <= window_size <= 30):
            raise ValueError("window_size must be between 2 and 30")
        self.window_size = window_size
        self.history: deque[str] = deque(maxlen=window_size)

    def add(self, fen: str) -> None:
        if not fen or not fen.strip():
            raise ValueError("fen cannot be empty")
        self.history.append(fen)

    def stable(self) -> str | None:
        if len(self.history) < self.window_size:
            return None
        counter = Counter(self.history)
        winner, count = counter.most_common(1)[0]
        if count >= (self.window_size // 2) + 1:
            return winner
        return None

