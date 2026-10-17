# Testing and QA

## Automated tests

```powershell
.\.venv\Scripts\python -m pytest tests -q
```

```bash
.venv/bin/python -m pytest tests -q          # macOS / Linux
```

No camera needed. 33 tests:

| File | Covers |
|---|---|
| `tests/test_calibration.py` | Outer corners extrapolated from the 49 inner corners (within 2 px), never mirroring (also on a mirrored image), corner ordering, manual corners, rotation, save/load, wrong frame size |
| `tests/test_stream.py` | Frame size names, the frame reader keeps only the newest frame |
| `tests/test_reader.py` | The change map sees a black piece on a dark square (texture), ignores a square edge nudged by 1–2 px, and an exposure fit thrown off by a hand doesn't shift every pixel |
| `tests/test_matcher.py` | A 26-ply opening with captures and kingside castling, queenside castling, en passant, promotion, still frames giving no move, a missed move leaving unexplained change, a faint black-on-dark e7-e5 still scoring once the board's contrast is known |
| `tests/test_replay.py` | Whole games recorded as sessions and replayed end to end: a 16-move game (a "hand" between moves, ~1 fps, exposure drift), impatient players, the camera bumped three times mid-game, moves taken back by hand, a dark board where the start calibration must be chosen, a hand resting beside a corner, a sleeve beside rank 8, a sleeve lying over h1, and the Board window's dots refreshing every move |
| `tests/test_engine.py` | Stockfish found from `engine.path`, else from `PATH` when the configured path is from another computer |

The camera scenes come from `tests/scene.py`: it draws a board and pieces through the same camera
model the reader uses and adds noise and exposure changes. Two cameras: a low side view (harder,
because pieces hide each other more) and a near-overhead one like the real mount, where pieces at
the edge lean out past the board. Options give a textured table, darker squares and fainter piece
outlines, to imitate the real board. The tests for each live bug fail on the code before its fix.

## Replay harness

Every `play` session saves its frames in `sessions/<date>_<time>/`. Replay runs them through the
same code and, given the moves that were really played, scores the result:

```powershell
.\.venv\Scripts\python run.py replay sessions\<date>_<time> --pgn tests\data\test_game.pgn
```

```
 ply    read   truth tier   margin
   1    e2e4    e2e4    1     3236 ok
   ...
auto-accept rate 100% | wrong accepts 0 | tiers {1: 1, 2: 0, 3: 0, 4: 0} | read 1/16
```

- **auto-accept rate**: moves accepted without asking (tiers 1–3).
- **wrong accepts**: moves accepted without asking that were wrong. **The target is 0**; the command
  exits with code 2 otherwise.
- When it would ask you (tier 4), the answer is taken from the PGN. After a wrong accept the move is
  corrected so the rest of the game still counts.
- `--llm` allows tier 3 (costs API calls; off by default).
- Replay PGNs are written to `<session>/replay/`, not `exports/`.

Use this after changing any threshold: replay the recorded games and check that wrong accepts stay
at 0.

A PGN can't describe moves taken back by hand, so for games with takebacks compare the final board
and the log instead: each `Move N: ...` line and each `was taken back` line. The figures below were
measured that way, frame by frame: on every frame where the board was still, is the program's
position the one really on the board?

## The test game

[tests/data/test_game.pgn](../tests/data/test_game.pgn):
`1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.O-O Nf6 5.d3 d6 6.Bg5 h6 7.Bxf6 Qxf6 8.Nc3 O-O`.
Pawn moves, knights, bishops, captures by bishop and queen, castling for both sides.

To record it: start position on the board, `run.py play --start`, play the moves pausing ~3 s after
each, press Q. Then replay the newest `sessions\` folder against the PGN.

## QA results

All on the real board and camera unless marked synthetic.

| Date | Check | Result |
|---|---|---|
| 2026-10-17 | YOLO piece model on a starting-position frame | 9–15 of 32 pieces found, mostly labelled "knight". Reason for the change-based design |
| 2026-10-17 | Image orientation | Raw image was vertically mirrored (printed digits read backwards); fixed with `vflip` |
| 2026-10-17 | Automatic calibration with pieces set up, overhead view | Fit error 0.79–0.85 px (of 100 per square), every run |
| 2026-10-17 | Orientation from piece colours | Correct every run, contrast 80–83 (wrong rotations −85 or ~0) |
| 2026-10-17 | 47 still frames (45 s) | 0 false moves; every move scored about −2,700 |
| 2026-10-17 | Occupancy sync check on the starting position | 0 mismatches |
| 2026-10-17 | Stockfish 18 | Running; +0.40, best e4 from the start |
| 2026-10-17 | First real move, 1.e4 | Read correctly (top choice) but asked instead of auto-accepting: 68% "unexplained" change from auto-exposure and the vinyl crease. Fixed with exposure gain correction and blob-based unexplained change. Replay of the same frames: **tier 1, margin 3,236, 4% unexplained** |
| 2026-10-17 | Synthetic 16-move game, ~1 fps | 16/16 read, 0 wrong, 94% automatic (14 tier 1, 1 tier 2, 1 asked) |
| 2026-10-17 | Synthetic, impatient players | 16/16 read, 0 wrong |
| 2026-10-17 | Board moved after the camera was moved between runs | Saved calibration 25–80 px off (a square is ~55 px); no move read at all. Fixed: the board is found again at each start, and camera shifts are followed |
| 2026-10-17 | Game with four moves taken back by hand (d5, d6, Nc6, h5) | Before: board state wrong on 196 of 290 still frames. After: **0 of 290**; each takeback undone automatically |
| 2026-10-17 | 1.e4 e5 (black pawn, dark square to dark square) | Before: e5 seen but silently dropped, 43 of 75 still frames wrong. After: **tier 1**, 0 wrong |
| 2026-10-17 | Real game, 10 plies (e4 e5 Bc4 Nf6 Nf3 d6 Nc3 Be7 O-O c5) | Before: c5 and e5 asked; a hand by the h1 corner read as Kh1 at tier 1. After: **0 of 527** still frames wrong, c5 tier 1, no Kh1 |
| 2026-10-17 | Real game, 20 plies (e4 e5 Nf3 Qf6 Bd3 d6 O-O Bg4 Nc3 d5 Nxd5 Nd7 b3 O-O-O Bb2 Ba3 Bxa3 Ne7 h3 Rhf8, with one takeback) | Before: a sleeve over the h1 corner read as Rg1 and three times as Kh1. After: **0 of 874** still frames wrong; the only questions came with a hand at the board |
| 2026-10-17 | A session with a cable dangling over the board, nothing moved | Before: "h3" read. After: nothing read |
| — | **The 16-move test game, recorded and replayed against `test_game.pgn`** | **Pending** |

Bugs found by QA and fixed: the frame reader thread crashed when two frames arrived within the
Windows clock tick; automatic calibration gave up before this slow camera's first frame; lighting
drift triggered a "which move?" question; a pending uncertain move was dropped when the next move
started; the PGN save crashed the game when its folder disappeared; a stale calibration was reused
after the camera moved; the exposure fit kept a wrong offset after clipping its gain, so a hand in
the reference frame made the whole board look changed; the frame from question time (a hand still
on the board) became the reference after an answer; a faint but clear move was dropped as
lighting; a hand or sleeve at the board's edge blocked real moves or passed as one; the Board
window's dots only updated every 10 plies; Stockfish wasn't found when `config.yaml` held a path
from another computer.
