"""End-to-end: synthetic recorded session -> replay harness -> zero wrong accepts."""

import csv
import json

import chess
import chess.pgn
import cv2
import numpy as np

from chessreader.replay import run_replay
from chessreader.settings import Settings
from tests.scene import FRAME, make_camera, render, table_texture

GAME = "e4 e5 Nf3 Nc6 Bc4 Bc5 O-O Nf6 d3 d6 Bg5 h6 Bxf6 Qxf6 Nc3 O-O".split()


def write_session(tmp_path, moves, seed=0, hand_frames=2, still_frames=3, bumps=None):
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
        emit(render(cam, cal, board, rng, table=table), "WAIT_START")
    for i, san in enumerate(moves):
        move = board.parse_san(san)
        if bumps and i in bumps:
            dx, dy, degrees = bumps[i]
            turn = np.vstack([cv2.getRotationMatrix2D((FRAME[0] / 2, FRAME[1] / 2), degrees, 1.0), [0, 0, 1]])
            camera_shift = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]]) @ turn @ camera_shift
        # a "hand" sweeping in over the from-square, then the new position
        x, y = cam.project(np.array([[chess.square_file(move.from_square) + .5,
                                      7 - chess.square_rank(move.from_square) + .5, -1.0]]))[0]
        for k in range(hand_frames):
            img = render(cam, cal, board, rng, table=table)
            cv2.ellipse(img, (int(x) + 30 * k, int(y)), (70, 45), 20, 0, 360, (120, 150, 200), -1)
            emit(img, "PLAYING")
        board.push(move)
        for _ in range(still_frames):
            emit(render(cam, cal, board, rng, exposure=rng.uniform(-4, 4), table=table), "PLAYING")
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
