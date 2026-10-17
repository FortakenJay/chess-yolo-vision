"""End-to-end: synthetic recorded session -> replay harness -> zero wrong accepts."""

import csv
import json

import chess
import chess.pgn
import cv2
import numpy as np

from chessreader.replay import run_replay
from chessreader.settings import Settings
from tests.scene import make_camera, render

GAME = "e4 e5 Nf3 Nc6 Bc4 Bc5 O-O Nf6 d3 d6 Bg5 h6 Bxf6 Qxf6 Nc3 O-O".split()


def write_session(tmp_path, moves, seed=0, hand_frames=2, still_frames=3):
    cal, cam = make_camera()
    rng = np.random.default_rng(seed)
    board = chess.Board()
    folder = tmp_path / "session"
    (folder / "frames").mkdir(parents=True)
    (folder / "calibration.json").write_text(json.dumps(cal.to_json()))
    rows, t, n = [], 0.0, 0

    def emit(image, phase):
        nonlocal t, n
        name = f"{n:06d}.jpg"
        cv2.imwrite(str(folder / "frames" / name), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append([name, f"{t:.3f}", phase])
        t += 1.2  # the real camera delivers about one frame per second
        n += 1

    for _ in range(2):
        emit(render(cam, cal, board, rng), "WAIT_START")
    for i, san in enumerate(moves):
        move = board.parse_san(san)
        # a "hand" sweeping in over the from-square, then the new position
        x, y = cam.project(np.array([[chess.square_file(move.from_square) + .5,
                                      7 - chess.square_rank(move.from_square) + .5, -1.0]]))[0]
        for k in range(hand_frames):
            img = render(cam, cal, board, rng)
            cv2.ellipse(img, (int(x) + 30 * k, int(y)), (70, 45), 20, 0, 360, (120, 150, 200), -1)
            emit(img, "PLAYING")
        board.push(move)
        for _ in range(still_frames):
            emit(render(cam, cal, board, rng, exposure=rng.uniform(-4, 4)), "PLAYING")
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
