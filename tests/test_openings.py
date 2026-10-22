"""Opening book: the data file is sound, lookups name the right openings, PGNs get ECO/Opening tags."""

import csv
import io
import re

import chess
import chess.pgn
import pytest

from chessreader.openings import DEFAULT_BOOK, OpeningBook, default_book, load_book
from chessreader.storage import GameRecord


def board_after(moves: str, fen: str | None = None) -> chess.Board:
    board = chess.Board(fen) if fen else chess.Board()
    for san in moves.split():
        board.push_san(san)
    return board


@pytest.fixture(scope="module")
def book() -> OpeningBook:
    return default_book()


def test_every_line_is_legal_and_well_formed():
    with DEFAULT_BOOK.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) >= 300
    for row in rows:
        assert re.fullmatch(r"[A-E]\d\d", row["eco"]), row
        assert row["name"].strip() == row["name"] and row["name"], row
        game = chess.pgn.read_game(io.StringIO(row["pgn"]))
        assert game is not None and not game.errors, row
        moves = list(game.mainline_moves())
        assert len(moves) >= 1, row
        board = chess.Board()
        for move in moves:
            assert board.is_legal(move), row
            board.push(move)
    book = OpeningBook.load(DEFAULT_BOOK)
    assert book.lines == len(rows)


def test_first_moves_have_names(book):
    for moves, name in [
        ("e4", "King's Pawn Game"),
        ("d4", "Queen's Pawn Game"),
        ("c4", "English Opening"),
        ("Nf3", "Zukertort Opening"),
        ("e4 e5", "King's Pawn Game"),
        ("e4 c5", "Sicilian Defense"),
        ("e4 e6", "French Defense"),
        ("e4 c6", "Caro-Kann Defense"),
        ("d4 d5 c4", "Queen's Gambit"),
    ]:
        info = book.lookup(board_after(moves))
        assert info.name == name, moves
        assert info.in_book and info.left_book_ply is None


def test_known_lookups(book):
    ruy = book.lookup(board_after("e4 e5 Nf3 Nc6 Bb5"))
    assert ruy.name == "Ruy Lopez" and ruy.eco == "C60"
    berlin = book.lookup(board_after("e4 e5 Nf3 Nc6 Bb5 Nf6"))
    assert berlin.name == "Ruy Lopez: Berlin Defense" and berlin.eco == "C65"
    kid = book.lookup(board_after("d4 Nf6 c4 g6"))
    assert kid.name.startswith("King's Indian Defense") and kid.eco.startswith("E6")
    grunfeld = book.lookup(board_after("d4 Nf6 c4 g6 Nc3 d5"))
    assert grunfeld.name == "Grünfeld Defense" and grunfeld.eco == "D80"
    najdorf = book.lookup(board_after("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6"))
    assert najdorf.name == "Sicilian Defense: Najdorf Variation" and najdorf.eco == "B90"
    giuoco = book.lookup(board_after("e4 e5 Nf3 Nc6 Bc4 Bc5"))
    assert giuoco.name == "Italian Game: Giuoco Piano" and giuoco.eco == "C50"
    nimzo = book.lookup(board_after("d4 Nf6 c4 e6 Nc3 Bb4"))
    assert nimzo.name == "Nimzo-Indian Defense" and nimzo.eco == "E20"


def test_transposition_gets_the_same_name(book):
    a = book.lookup(board_after("d4 d5 c4 c6 Nf3 Nf6 Nc3 e6"))
    b = book.lookup(board_after("d4 d5 c4 e6 Nc3 Nf6 Nf3 c6"))
    c = book.lookup(board_after("Nf3 d5 d4 Nf6 c4 e6 Nc3 c6"))
    assert a.name == b.name == c.name == "Semi-Slav Defense"
    assert a.eco == b.eco == c.eco == "D43"
    assert c.in_book
    eng1 = book.lookup(board_after("c4 e5 Nc3 Nf6"))
    eng2 = book.lookup(board_after("c4 Nf6 Nc3 e5"))
    assert eng1.name == eng2.name and eng1.eco == eng2.eco == "A22"


def test_leaving_book_keeps_the_last_name(book):
    board = board_after("e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6")
    named = book.lookup(board)
    assert named.in_book and named.name.startswith("Italian Game")
    board.push_san("Qb3")  # ply 9: not theory
    board.push_san("Qe7")
    info = book.lookup(board)
    assert not info.in_book
    assert info.left_book_ply == 9
    assert info.name == named.name and info.eco == named.eco
    assert info.book_moves == []


def test_book_moves_from_the_start(book):
    info = book.lookup(chess.Board())
    assert info.in_book and info.name is None and info.left_book_ply is None
    first_two = set(info.book_moves[:2])
    assert first_two == {chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")}
    sans = book.book_san(chess.Board(), limit=4)
    assert len(sans) == 4 and "e4" in sans and "d4" in sans
    after_e4_e5 = book.book_san(board_after("e4 e5"), limit=1)
    assert after_e4_e5 == ["Nf3"]


def test_custom_start_has_no_opening(book):
    board = board_after("e5", fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    info = book.lookup(board)
    assert info.name is None and info.eco is None and not info.in_book and info.book_moves == []
    assert book.book_san(board) == []


def test_lookup_does_not_change_the_board(book):
    board = board_after("e4 c5 Nf3")
    fen, stack = board.fen(), list(board.move_stack)
    book.lookup(board)
    book.book_san(board)
    assert board.fen() == fen and board.move_stack == stack


def headers_of(record: GameRecord) -> chess.pgn.Headers:
    return chess.pgn.read_game(io.StringIO(record.path.read_text(encoding="utf-8"))).headers


def test_pgn_gets_eco_and_opening_tags(tmp_path):
    record = GameRecord(tmp_path)
    assert "Opening" not in headers_of(record)
    for san in ["e4", "e5", "Nf3", "Nc6", "Bb5"]:
        record.add_move(record.board.parse_san(san), tier=1, margin=1.0)
    h = headers_of(record)
    assert h["ECO"] == "C60" and h["Opening"] == "Ruy Lopez"
    record.add_move(record.board.parse_san("a6"), tier=1, margin=1.0)
    assert headers_of(record)["Opening"] == "Ruy Lopez: Morphy Defense"
    record.undo()
    h = headers_of(record)
    assert h["ECO"] == "C60" and h["Opening"] == "Ruy Lopez"
    for _ in range(5):
        record.undo()
    h = headers_of(record)
    assert "ECO" not in h and "Opening" not in h


def test_pgn_still_saves_without_a_book(tmp_path, monkeypatch):
    def broken():
        raise OSError("no book")

    monkeypatch.setattr("chessreader.openings.default_book", broken)
    load_book.cache_clear()  # forget the book loaded by earlier tests
    record = GameRecord(tmp_path)
    record.add_move(chess.Move.from_uci("e2e4"), tier=1, margin=1.0)
    h = headers_of(record)
    assert "Opening" not in h and record.pgn().strip().endswith("*")
    assert "1. e4" in record.path.read_text(encoding="utf-8")
    load_book.cache_clear()  # later tests get the real book again
