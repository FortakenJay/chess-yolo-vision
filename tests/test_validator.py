from src.validator import is_legal_position


def test_known_legal_positions():
    legal = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/3P4/PPP2PPP/RNBQKBNR w KQkq - 0 3",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    ]
    assert all(is_legal_position(fen) for fen in legal)


def test_known_illegal_positions():
    illegal = [
        "",
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "4k3/8/8/8/8/8/8/4P3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
        "8/8/8/8/8/8/4K3/4K3 w - - 0 1",
    ]
    assert all(not is_legal_position(fen) for fen in illegal)

