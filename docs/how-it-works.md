# How it works

```
camera ─▶ stream.py ─▶ session.py ──────────────────────────────────────────────────────┐
                        │ gating.py      is the board still?                              │
                        │ reader.py      what changed since the last accepted position?    │
                        │ matcher.py     which legal move explains it best?   ◀─ geometry.py (camera model)
                        │ confirm.py     confident? re-read? ask?                          │
                        │ health.py      still in sync? board moved?                        │
                        ▼                                                                   │
              storage.py (PGN)   engine.py (Stockfish)   play.py (windows, keys)  ◀──────────┘
```

`session.py` holds all game logic and has no UI, so live play (`play.py`) and the replay
harness (`replay.py`) run exactly the same code.

## Why not just detect the pieces?

The first design ran the YOLO piece model (`models/pieces.pt`, 13 classes) on every frame and built
a FEN from the boxes. On this camera it found only **9–15 of 32 pieces** in the starting position,
and labelled most of them "knight". It was trained on Roboflow photos that look nothing like a dim,
green-tinted 800×600 ESP32 image. Lower thresholds or larger input sizes didn't help.

So the reader never asks "what piece is on e4?". It asks "of the legal moves, which one would make
the board change the way it just did?". That needs no piece recognition, works with any set, and
python-chess guarantees the result is legal. YOLO is used only to tell what a pawn promoted to.

## 1. Camera input — `stream.py`

- At startup the PC sets the camera over HTTP: frame size (by name, e.g. `SVGA`), JPEG quality,
  `vflip`/`hmirror`. It then reads `/status` and the size of a real image to confirm the change took.
- A background thread fetches frames continuously and keeps **only the newest one**, so the game
  logic never works on stale frames. Two modes:
  - `capture` (default): polls `/capture`. Works even if a browser has the camera page open.
  - `stream`: reads the MJPEG stream on port 81 directly (no FFmpeg, which gave up while probing
    the slow stream).
- On any failure it reconnects with backoff. Before reconnecting it checks the network and logs
  whether the PC is on `ChessCam` at all.
- Camera requests bypass any HTTP proxy (campus networks often set one).

## 2. Calibration — `calibration.py`

Goal: a homography `H` from camera pixels to an 800×800 top-down board (100 px per square).

- **Empty board**: OpenCV finds all 7×7 inner corners. `H` is fitted by least squares to all 49
  points; the outer corners come from mapping the board edge (one square outward) back through `H`.
- **Starting position**: pieces cover the corners on ranks 1–2 and 7–8, but ranks 3–6 are empty. The
  7×3 inner-corner block is found and mapped to ranks 3–6, which pins down the whole board.
- Several OpenCV detectors are tried; a result is only accepted if its corners lie on a near-perfect
  grid (mean error under 3% of a square). The newer `findChessboardCornersSB` sometimes reports
  success with scrambled corners when pieces are on the board; this check catches that.
- **Never mirrored**: corners are always ordered clockwise in the image, so the top-down view can be
  rotated wrongly but never mirrored. Rotation is fixed in step 4.
- Fallback: click the 4 outer corners (with a zoom loupe). Saved to `calib/` as JSON plus reference
  images (raw frame, top-down view, edge map).

## 3. Camera model — `geometry.py`

From `H` the program recovers a 3D camera so it can predict where a piece *of a given height*
appears in the image.

- **Focal length** from the homography itself. With the principal point at the image centre and
  `h1`, `h2` the first two columns of the board→image homography, the board's axes must be
  perpendicular and equally long, which gives
  `f² = −(h1x·h2x + h1y·h2y) / (h1z·h2z)` (and a second estimate from the equal-length condition).
  Seen nearly straight from above this is ill-conditioned; then the OV2640's ~66° diagonal field of
  view is assumed.
- **Pose** with `cv2.solvePnP` on the 81 grid points.
- **Pieces** are modelled as tapered cylinders (base radius 0.36, top radius 0.24 squares) with heights
  in squares: pawn 0.85, knight 1.2, bishop 1.4, rook 1.0, queen 1.55, king 1.75 (`reader.height_scale`
  scales them). Each piece's outline in the image is the convex hull of its projected base and top.
- **Rendering a position** paints these outlines far-to-near into a label image, so nearer pieces
  hide farther ones. Each pixel stores which piece (square + type + colour) is visible there.
- **Orientation**: for each of the 4 possible rotations the starting position is rendered and the
  brightness of pixels predicted to be white pieces is compared with black ones. The right rotation
  scores about +80 (gray levels); the wrong ones about −80 or near 0.

## 4. Motion gate — `gating.py`

Each frame is cropped to the board region, shrunk to 240 px wide, blurred, and compared with the
previous one. If more than 0.4% of the board pixels changed by more than 18 gray levels, the board
is **moving** (a hand, an arm). It becomes **stable** after 0.8 s and at least 2 frames without
motion. The board is read once, at the moment it becomes stable.

## 5. Change map — `reader.py`

The new still frame is compared with the frame saved when the last move was accepted:

1. Both frames are cropped to the board region, shrunk to half size, blurred, converted to Lab.
2. **Exposure correction**: the camera's automatic exposure and white balance make bright pixels
   change more than dark ones. Per channel, `current ≈ gain · reference + offset` is fitted to the
   pixels that look unchanged (two passes, dropping outliers), and the reference is mapped onto the
   current exposure.
3. Per pixel, `D = max(|ΔL|, 1.5 · |Δ(a,b)|)`.
4. `D` becomes a change probability `b`, ramping from 0 at `t_low` (10) to 1 at `t_high` (28).

## 6. Move matching — `matcher.py`

For the current position and each legal move, the position is rendered before and after the move.
Pixels whose visible piece differs form the move's **predicted mask M**. Captures, castling (four
squares) and en passant (three) fall out of this automatically.

Each move is scored by the log-likelihood ratio of "this move happened" against "nothing happened":

```
score(move) = Σ over pixels p in M of   b(p) · ln(q_in / q_out)  +  (1 − b(p)) · ln((1 − q_in) / (1 − q_out))
```

- `q_in` = 0.55: chance a pixel inside a moved piece's outline visibly changes (outlines are
  approximate, and a white piece on a white square changes little).
- `q_out` = 0.03: noise rate.
- A changed pixel inside the mask earns ln(0.55/0.03) ≈ +2.9; an unchanged pixel inside it costs
  ln(0.45/0.97) ≈ −0.77.
- Scores are in full-resolution pixel units, so thresholds don't depend on the work scale.

Under-promotions are scored as queen promotions (they look identical to the camera) and decided
afterwards.

The result is the ranked list, the **margin** (best minus runner-up), and the **unexplained**
share: change outside the best move's mask that forms piece-sized blobs. Speckle and thin lines are
removed with a morphological opening about a third of a square wide, so a vinyl board's crease
shifting doesn't count, but a second moved or bumped piece does.

Typical numbers on the real board: a clear pawn move scores about 3,100–3,500 with a margin over
3,000. Still frames score about −2,700 for every move.

## 7. Decisions and tiers — `session.py`, `confirm.py`

When the board becomes still:

1. Total change under `min_change_px` (900) → nothing happened.
2. More than 35% of the board changed → something covers the board; wait.
3. No move scores at least `min_score` (1,000) → if the change is small (under 6,000 px) it's
   lighting drift and is ignored; otherwise you're asked, with **X** available.
4. **Tier 1**: score ≥ 1,000, margin ≥ 1,500 and unexplained ≤ 35% → accepted.
5. **Tier 2**: read 3 more still frames and average each move's score. Margin ≥ 800 → accepted. If
   the next move starts before the re-reads finish, it decides with the reads it has.
6. **Tier 3** (off by default): before/after photos and the top 3 moves go to a vision LLM (Anthropic
   API, `claude-sonnet-5-5`), which must answer with one of the listed moves.
7. **Tier 4**: the top 3 are drawn as numbered arrows; you press 1–3 or type it. The reference
   becomes the frame from when the question was asked, so a move made in the meantime isn't lost,
   and it's read right after you answer.

Stockfish is never consulted here, so a blunder is recorded as played.

## 8. Health checks — `health.py`

- **Sync** (every 10 moves): the top-down view's square centres are classified as empty / white /
  black, using colour templates learned from the starting position (empty light, empty dark, white
  piece, black piece). Squares whose centre is hidden by another piece are skipped. A square that
  confidently disagrees with the game on two checks in a row is flagged in red, with Z/T/X offered.
  Black pieces on dark squares are often too close to call; those are skipped rather than flagged.
- **Drift** (every 10 s while still): edges of the top-down view are compared with the calibration
  edge map. If under 45% of the reference edges are still within 4 px on 3 checks in a row, the game
  pauses and asks you to recalibrate (**C**).

## 9. Promotion

When the chosen move is a promotion, the YOLO model runs on that frame. A detection counts if its
**bottom-centre** (where the piece stands, not the middle of its box) maps through `H` onto the
promotion square and its colour matches. The most confident piece type wins; otherwise a queen.

## 10. Output — `storage.py`, `engine.py`, `play.py`

- **PGN**: standard tags, the moves, and a comment per move with the time and tier. Written to a
  temporary file and swapped in, after every move. A failed write is logged and retried; it never
  stops the game.
- **Stockfish** runs in its own thread: 0.5 s per position, 3 lines, win/draw/loss when the engine
  supports it. Only the newest position is analysed. Results go to the Board window.
- **Session recording**: every frame plus the moves and calibration, for `replay`.
