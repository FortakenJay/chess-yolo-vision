"""End-to-end: synthetic recorded session -> replay harness -> zero wrong accepts."""

import csv
import json

import chess
import chess.pgn
import cv2
import numpy as np

from chessreader.replay import run_replay
from chessreader.settings import Settings
from tests.scene import FRAME, OUTER, OVERHEAD, make_camera, render, table_texture

GAME = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O", "Nf6", "d3", "d6", "Bg5", "h6", "Bxf6", "Qxf6", "Nc3", "O-O"]


def write_session(
    tmp_path,
    moves,
    seed=0,
    hand_frames=2,
    still_frames=3,
    bumps=None,
    dark_square=(40, 70, 45),
    outline=(110, 110, 110),
):
    """bumps: {move index: (dx, dy, degrees)}, the camera shifting while that move is made."""
    cal, cam = make_camera()
    rng = np.random.default_rng(seed)
    table = table_texture(seed) if bumps else None
    camera_shift = np.eye(3)
    board = chess.Board()
    folder = tmp_path / "session"
    (folder / "frames").mkdir(parents=True)
    (folder / "calibration.json").write_text(json.dumps(cal.to_json()))
    rows, t, n = [], 0.0, 0

    def emit(image, phase):
        nonlocal t, n
        if bumps:
            image = cv2.warpPerspective(image, camera_shift, FRAME, borderMode=cv2.BORDER_REPLICATE)
        name = f"{n:06d}.jpg"
        cv2.imwrite(str(folder / "frames" / name), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append([name, f"{t:.3f}", phase])
        t += 1.2  # the real camera delivers about one frame per second
        n += 1

    for _ in range(2):
        emit(render(cam, cal, board, rng, table=table, dark_square=dark_square, outline=outline), "WAIT_START")
    for i, san in enumerate(moves):
        move = board.peek() if san == "takeback" else board.parse_san(san)
        if bumps and i in bumps:
            dx, dy, degrees = bumps[i]
            turn = np.vstack([cv2.getRotationMatrix2D((FRAME[0] / 2, FRAME[1] / 2), degrees, 1.0), [0, 0, 1]])
            camera_shift = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]]) @ turn @ camera_shift
        # a "hand" sweeping in over the from-square, then the new position
        x, y = cam.project(
            np.array([[chess.square_file(move.from_square) + 0.5, 7 - chess.square_rank(move.from_square) + 0.5, -1.0]])
        )[0]
        for k in range(hand_frames):
            img = render(cam, cal, board, rng, table=table, dark_square=dark_square, outline=outline)
            cv2.ellipse(img, (int(x) + 30 * k, int(y)), (70, 45), 20, 0, 360, (120, 150, 200), -1)
            emit(img, "PLAYING")
        if san == "takeback":  # the player puts the last move back by hand
            board.pop()
        else:
            board.push(move)
        for _ in range(still_frames):
            emit(
                render(
                    cam,
                    cal,
                    board,
                    rng,
                    exposure=rng.uniform(-4, 4),
                    table=table,
                    dark_square=dark_square,
                    outline=outline,
                ),
                "PLAYING",
            )
    with open(folder / "frames.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    game = chess.pgn.Game.from_board(board)
    pgn = tmp_path / "truth.pgn"
    pgn.write_text(str(game), encoding="utf-8")
    return folder, pgn


def test_replay_reads_every_move_without_wrong_accepts(tmp_path):
    folder, pgn = write_session(tmp_path, GAME, still_frames=5)
    settings = Settings()
    settings.storage.exports_dir = str(tmp_path / "exports")
    settings.engine.enabled = False
    summary = run_replay(settings, folder, pgn)
    assert summary["wrong_accepts"] == 0
    assert summary["moves_read"] == len(GAME)
    assert summary["auto_accept_rate"] >= 0.9


def test_quick_players_never_cause_wrong_accepts(tmp_path):
    """Only 3 still frames between moves: re-reads get cut short, but nothing may be misread."""
    folder, pgn = write_session(tmp_path, GAME, seed=5, still_frames=3)
    settings = Settings()
    settings.storage.exports_dir = str(tmp_path / "exports")
    settings.engine.enabled = False
    summary = run_replay(settings, folder, pgn)
    assert summary["wrong_accepts"] == 0
    assert summary["moves_read"] == len(GAME)


def test_camera_bumped_mid_game_is_followed(tmp_path):
    """The camera shifts (up to about a square) while moves are made: no recalibration needed."""
    bumps = {3: (25, -12, 1.5), 7: (-40, 20, -2.0), 11: (8, 30, 0.0)}
    folder, pgn = write_session(tmp_path, GAME, seed=3, still_frames=5, bumps=bumps)
    settings = Settings()
    settings.storage.exports_dir = str(tmp_path / "exports")
    settings.engine.enabled = False
    summary = run_replay(settings, folder, pgn)
    assert summary["wrong_accepts"] == 0
    assert summary["moves_read"] == len(GAME)
    assert summary["auto_accept_rate"] >= 0.9


def test_moves_put_back_by_hand_are_taken_back(tmp_path):
    """The live game that kept a move the player had put back: the board state must follow the hands."""
    from chessreader.calibration import Calibration
    from chessreader.replay import frames_from
    from chessreader.session import GameSession, Phase

    moves = ["e4", "d5", "takeback", "Nc6", "takeback", "h5", "takeback", "c5", "Nf3"]
    folder, _ = write_session(tmp_path, moves, seed=7, still_frames=5)
    settings = Settings()
    settings.storage.exports_dir = str(tmp_path / "exports")
    settings.engine.enabled = False
    frames = list(frames_from(folder))
    cal = Calibration.from_json(json.loads((folder / "calibration.json").read_text()))
    session = GameSession(settings, cal)
    assert session.start_game(frames[0][0]) is None
    for image, ts, _ in frames[1:]:
        session.process(image, ts)
        assert session.phase != Phase.ASK_USER, f"asked about {session.choices}"
    assert [m.uci() for m in session.board.move_stack] == ["e2e4", "c7c5", "g1f3"]


def test_dark_board_start_keeps_the_calibration_that_fits(tmp_path):
    """On a dark board the start position's middle ranks are found, but tall pieces can hide corners
    and snap the grid a rank off. Start must keep whichever calibration lines the pieces up."""
    folder, pgn = write_session(
        tmp_path,
        ["e4", "g6", "d4", "Bg7", "Nf3", "Bf6"],
        seed=4,
        still_frames=5,
        dark_square=(34, 36, 38),
        outline=(48, 48, 50),
    )
    settings = Settings()
    settings.storage.exports_dir = str(tmp_path / "exports")
    settings.engine.enabled = False
    summary = run_replay(settings, folder, pgn)
    assert summary["wrong_accepts"] == 0
    assert summary["moves_read"] == 6
    assert summary["by_tier"][4] == 0


def _session_with_hand_off_board(tmp_path, at, radius, moves_before, move_during=None, outer=OUTER, **look):
    """A game where a hand comes to rest just beyond the board edge (and stays) after `moves_before`."""
    cal, cam = make_camera(outer)
    rng = np.random.default_rng(11)
    board = chess.Board()
    frames, t = [], 0.0
    x, y = cam.project(np.array([(*at, 0.0)]))[0]

    def add(hand: bool, n: int, sweep_over: str | None = None):
        nonlocal t
        for _ in range(n):
            img = render(cam, cal, board, rng, **look)
            if hand:
                cv2.ellipse(img, (int(x), int(y)), radius, 30, 0, 360, (120, 150, 200), -1)
            if sweep_over:  # the moving hand, so the motion gate sees the move being made
                sq = chess.parse_square(sweep_over)
                hx, hy = cam.project(np.array([[chess.square_file(sq) + 0.5, 7 - chess.square_rank(sq) + 0.5, -1.0]]))[
                    0
                ]
                cv2.ellipse(img, (int(hx), int(hy)), (70, 45), 20, 0, 360, (120, 150, 200), -1)
            frames.append((img, t))
            t += 1.2

    add(False, 3)
    for san in moves_before:
        board.push_san(san)
        add(False, 5)
    add(True, 6)  # the hand arrives and rests
    if move_during:
        move = board.parse_san(move_during)
        add(True, 2, sweep_over=chess.square_name(move.from_square))
        board.push(move)
        add(True, 6)
    return cal, frames


def _play(settings, cal, frames):
    from chessreader.session import GameSession

    session = GameSession(settings, cal)
    assert session.start_game(frames[0][0]) is None
    for image, ts in frames[1:]:
        session.process(image, ts)
    return session


def test_hand_resting_by_a_corner_is_not_a_move(tmp_path):
    """Live: a hand resting beside h1 (reaching for the laptop) was read as Kh1."""
    settings = Settings()
    settings.engine.enabled = False
    settings.storage.exports_dir = str(tmp_path / "exports")
    cal, frames = _session_with_hand_off_board(
        tmp_path, (8.45, 7.7), (40, 30), ["e4", "c5", "Nf3", "d6", "Bc4", "Nf6", "O-O", "a6"], outer=OVERHEAD
    )
    session = _play(settings, cal, frames)
    assert [m.uci() for m in session.board.move_stack] == [
        "e2e4",
        "c7c5",
        "g1f3",
        "d7d6",
        "f1c4",
        "g8f6",
        "e1g1",
        "a7a6",
    ]  # and no Kh1


def test_sleeve_beside_the_board_does_not_block_a_clean_move(tmp_path):
    """Live: an arm resting on the table beside rank 8 made every black move 'unexplained'."""
    settings = Settings()
    settings.engine.enabled = False
    settings.storage.exports_dir = str(tmp_path / "exports")
    cal, frames = _session_with_hand_off_board(
        tmp_path,
        (2.5, -0.55),
        (42, 24),
        ["e4"],
        move_during="c5",
        outer=OVERHEAD,
        dark_square=(34, 36, 38),
        outline=(48, 48, 50),
    )
    session = _play(settings, cal, frames)
    assert [m.uci() for m in session.board.move_stack] == ["e2e4", "c7c5"]


def test_sleeve_over_a_corner_square_is_not_a_move(tmp_path):
    """Live: a sweater sleeve lying over h1 (king on g1) was read as Kh1; the king never left g1."""
    settings = Settings()
    settings.engine.enabled = False
    settings.storage.exports_dir = str(tmp_path / "exports")
    cal, frames = _session_with_hand_off_board(
        tmp_path, (8.0, 8.0), (46, 34), ["e4", "c5", "Nf3", "d6", "Bc4", "Nf6", "O-O", "a6"], outer=OVERHEAD
    )
    session = _play(settings, cal, frames)
    assert [m.uci() for m in session.board.move_stack] == [
        "e2e4",
        "c7c5",
        "g1f3",
        "d7d6",
        "f1c4",
        "g8f6",
        "e1g1",
        "a7a6",
    ]


def test_board_dots_follow_every_move(tmp_path):
    """The Board window's per-square dots showed up only at the 10-ply health check, then froze."""
    from chessreader.calibration import Calibration
    from chessreader.replay import frames_from
    from chessreader.session import GameSession

    folder, _ = write_session(tmp_path, ["e4", "c5"], seed=2, still_frames=5)
    settings = Settings()
    settings.engine.enabled = False
    settings.storage.exports_dir = str(tmp_path / "exports")
    frames = list(frames_from(folder))
    session = GameSession(settings, Calibration.from_json(json.loads((folder / "calibration.json").read_text())))
    assert session.start_game(frames[0][0]) is None
    assert session.occupancy is not None
    seen = []
    for image, ts, _ in frames[1:]:
        before = len(session.moves)
        session.process(image, ts)
        if len(session.moves) != before:
            seen.append(session.occupancy.copy())
    assert len(seen) == 2 and not np.array_equal(seen[0], seen[1], equal_nan=True)


def test_pawn_stood_back_up_is_not_a_move(tmp_path):
    """Live: the b2 pawn lay on its side at the start; standing it up asked '1) b4 2) b3 3) h3'."""
    from chessreader.session import GameSession, Phase

    cal, cam = make_camera(OVERHEAD)
    rng = np.random.default_rng(5)
    board = chess.Board()
    tipped = board.copy()
    tipped.remove_piece_at(chess.B2)
    x, y = cam.project(np.array([[1.5, 6.5, 0.0]]))[0]
    frames, t = [], 0.0

    def add(b, n, lying=False, hand_over=None):
        nonlocal t
        for _ in range(n):
            img = render(cam, cal, b, rng, light_square=(248, 250, 248), dark_square=(70, 72, 74))
            if lying:  # the pawn on its side
                cv2.ellipse(img, (int(x) + 6, int(y)), (24, 10), 15, 0, 360, (215, 220, 220), -1)
                cv2.ellipse(img, (int(x) + 6, int(y)), (24, 10), 15, 0, 360, (110, 110, 110), 1)
            if hand_over is not None:
                hx, hy = cam.project(np.array([[*hand_over, -1.0]]))[0]
                cv2.ellipse(img, (int(hx), int(hy)), (70, 45), 20, 0, 360, (120, 150, 200), -1)
            frames.append((img, t))
            t += 1.2

    add(tipped, 4, lying=True)
    add(tipped, 2, lying=True, hand_over=(1.5, 6.5))  # the hand picks it up...
    add(board, 6)  # ...and it stands on b2 again
    add(board, 2, hand_over=(4.5, 6.5))
    board.push_san("e4")
    add(board, 6)

    settings = Settings()
    settings.engine.enabled = False
    settings.storage.exports_dir = str(tmp_path / "exports")
    session = GameSession(settings, cal)
    assert session.start_game(frames[0][0]) is None
    for image, ts in frames[1:]:
        session.process(image, ts)
        assert session.phase != Phase.ASK_USER, f"asked about {session.choices}"
    assert [m.uci() for m in session.board.move_stack] == ["e2e4"]
