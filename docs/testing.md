# Testing and QA

## Automated tests

```powershell
.\.venv\Scripts\python -m pytest tests -q
```

No camera needed. 19 tests:

| File | Covers |
|---|---|
| `tests/test_calibration.py` | Outer corners extrapolated from the 49 inner corners (within 2 px), never mirroring (also on a mirrored image), corner ordering, manual corners, rotation, save/load, wrong frame size |
| `tests/test_stream.py` | Frame size names, the frame reader keeps only the newest frame |
| `tests/test_matcher.py` | A 26-ply opening with captures and kingside castling, queenside castling, en passant, promotion, still frames giving no move, a missed move leaving unexplained change |
| `tests/test_replay.py` | A whole 16-move game recorded as a session (with a "hand" between moves, ~1 fps, exposure drift) replayed end to end: every move read, zero wrong accepts. Plus a variant with impatient players (only 3 still frames between moves) |

The camera scenes come from `tests/scene.py`: it draws a board and pieces through the same camera
model the reader uses, adds noise and exposure changes, and uses a low side view (harder than the
real overhead view, because pieces hide each other more).

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
| — | **Full real test game** | **Pending** |

Bugs found by QA and fixed: the frame reader thread crashed when two frames arrived within the
Windows clock tick; automatic calibration gave up before this slow camera's first frame; lighting
drift triggered a "which move?" question; a pending uncertain move was dropped when the next move
started; the PGN save crashed the game when its folder disappeared.
