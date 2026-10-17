"""Change map details that the real camera showed: black on black, nudged edges, exposure fits."""

import cv2
import numpy as np

from chessreader.reader import ChangeReader

SIZE = 200


def frame_with(patch: np.ndarray | None = None, shift: int = 0) -> np.ndarray:
    """A gray scene with a dark square edge down the middle and an optional patch at the center."""
    img = np.full((SIZE, SIZE, 3), 220, np.uint8)
    img[:, SIZE // 2 + shift:] = 75  # dark square
    if patch is not None:
        h, w = patch.shape[:2]
        y, x = SIZE // 2 - h // 2, 3 * SIZE // 4 - w // 2
        img[y:y + h, x:x + w] = patch
    return img


def black_piece_on_dark(size: int = 40) -> np.ndarray:
    """Same average brightness as the dark square, but a rim and a highlight like a real piece."""
    yy, xx = np.mgrid[:size, :size] - size / 2
    r = np.hypot(yy, xx)
    piece = np.full((size, size), 75.0)
    piece[r < size / 2 - 2] = 68.0
    piece[(r > size / 2 - 6) & (r < size / 2 - 3)] = 100.0   # rim catching the light
    piece[r < 4] = 105.0                                       # top highlight
    return np.repeat(np.clip(piece, 0, 255).astype(np.uint8)[..., None], 3, axis=2)


def change_px(reader: ChangeReader, a: np.ndarray, b: np.ndarray) -> float:
    return float(reader.change_probability(reader.prepare(a), reader.prepare(b)).sum())


def test_black_piece_on_dark_square_registers():
    piece = black_piece_on_dark()
    assert abs(float(piece.mean()) - 75) < 8  # the case brightness alone can't see
    with_texture = ChangeReader((0, 0, SIZE, SIZE), scale=0.5)
    without = ChangeReader((0, 0, SIZE, SIZE), scale=0.5, texture_weight=0.0)
    empty, occupied = frame_with(), frame_with(piece)
    assert change_px(with_texture, empty, occupied) > 2 * change_px(without, empty, occupied)
    assert change_px(with_texture, empty, occupied) > 60


def test_texture_adds_nothing_on_a_nudged_square_edge():
    """Edges are all contrast; a one-pixel camera nudge must not light up every one of them."""
    with_texture = ChangeReader((0, 0, SIZE, SIZE), scale=0.5)
    without = ChangeReader((0, 0, SIZE, SIZE), scale=0.5, texture_weight=0.0)
    for shift in (1, 2):
        a, b = frame_with(), frame_with(shift=shift)
        assert change_px(with_texture, a, b) <= 1.1 * change_px(without, a, b) + 5


def test_exposure_fit_thrown_off_does_not_shift_every_pixel():
    """A big bright blob (a hand) in the reference once clipped the gain but kept a bad offset."""
    reader = ChangeReader((0, 0, SIZE, SIZE), scale=0.5)
    ref = frame_with()
    cv2.circle(ref, (60, 60), 45, (250, 250, 250), -1)
    cur = frame_with()
    diff = reader.difference(reader.prepare(ref), reader.prepare(cur))
    untouched = np.ones(diff.shape, bool)
    cv2.circle(untouched.view(np.uint8), (30, 30), 30, 0, -1)
    assert float(np.median(diff[untouched])) < 5


def test_shaded_empty_square_at_the_edge_reads_empty():
    """Live: empty a6 at the board's shaded edge was as dark as a white piece, so it got a white dot."""
    import chess

    from chessreader.reader import OccupancyReader
    from tests.scene import OVERHEAD, make_camera, render

    cal, cam = make_camera(OVERHEAD)
    board = chess.Board()
    occ = OccupancyReader()
    warped = cal.warp(render(cam, cal, board, np.random.default_rng(0), noise=1.5))
    occ.fit(warped, cam, board)
    shaded = warped.copy()
    a6 = shaded[200:300, 0:100].astype(np.int16)  # row 2 = rank 6, column 0 = a-file
    shaded[200:300, 0:100] = np.clip(a6 - 18, 0, 255).astype(np.uint8)  # edge shading
    probs = occ.read(shaded, cam, board)
    assert int(np.argmax(probs[2, 0])) == 0  # empty
