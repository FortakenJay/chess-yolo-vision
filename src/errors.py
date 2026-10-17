"""Custom exception types for chess vision."""


class StreamConnectionError(RuntimeError):
    """Raised when stream cannot be reached/recovered."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"Stream connection error for {url}: {reason}")
        self.url = url
        self.reason = reason


class FENBuildError(ValueError):
    """Raised when a board map cannot be converted to FEN."""

    def __init__(self, rank: int | str, value: str) -> None:
        super().__init__(f"Failed to build FEN at rank={rank}: {value}")
        self.rank = rank
        self.value = value


class IllegalMoveError(ValueError):
    """Raised when a FEN delta cannot be parsed into a legal move."""

    def __init__(self, from_sq: str, to_sq: str, fen: str) -> None:
        super().__init__(f"Illegal move detected from {from_sq} to {to_sq} for {fen}")
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.fen = fen


class SupabaseInsertError(RuntimeError):
    """Raised when Supabase write fails after fallback archive."""

    def __init__(self, table: str, reason: str) -> None:
        super().__init__(f"Supabase insert failed for table={table}: {reason}")
        self.table = table
        self.reason = reason

