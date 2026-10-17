# How it works

```
camera ─▶ stream.py ─▶ session.py ──────────────────────────────────────────────────────┐
                        │ gating.py      is the board still?                              │
                        │ tracking.py    did the camera shift? follow it                   │
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
- **At every game start** the board is found again in the starting position, because the saved
  calibration is stale as soon as the camera has moved (live, it was 25–80 px off between runs, a
  square being ~55 px). Both the board found now and the saved calibration are oriented (step 3),
  and the one that lines the pieces up better wins. That guards against a detection that snapped
  a rank off when tall pieces hide corners; on the real board the fresh one wins, e.g. 87 against 50.

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
  scores about +85 (gray levels); the wrong ones about −85 or near 0. A stale calibration scores
  20–50.

## 4. Motion gate — `gating.py`

Each frame is cropped to the board region, shrunk to 240 px wide, blurred, and compared with the
previous one. If more than 0.4% of the board pixels changed by more than 18 gray levels, the board
is **moving** (a hand, an arm). It becomes **stable** after 0.8 s and at least 2 frames without
motion. The board is read once, at the moment it becomes stable.

## 4b. Camera tracking — `tracking.py`

Each time the board goes still (and right after you answer a question), ORB features in and
around the board (piece tops, printed coordinates, board edge, table) are matched between the
last accepted frame and the new one, and RANSAC fits the homography between them. If the board
corners moved more than `health.track_min_px` (2 px), the calibration and the reference frame are
moved with it, so a nudged or swaying camera needs no recalibration and a move made at the same
time is still read. Ambiguous matches are dropped (ratio test): the checker pattern repeats every
square, and a fit must not snap one square off. Shifts over 3 squares aren't trusted.

## 5. Change map — `reader.py`

The new still frame is compared with the frame saved when the last move was accepted:

1. Both frames are cropped to the board region, shrunk to half size, blurred, converted to Lab.
2. **Exposure correction**: the camera's automatic exposure and white balance make bright pixels
   change more than dark ones. Per channel, `current ≈ gain · reference + offset` is fitted to the
   pixels that look unchanged (two passes, dropping outliers), and the reference is mapped onto the
   current exposure. If the gain has to be clipped (a hand in one frame skews the fit), the offset
   is refitted for the clipped gain.
3. Per pixel, `D = max(|ΔL|, 1.5 · |Δ(a,b)|, 5 · Δtexture)`. **Texture** is the local contrast
   (brightness std in a ~14 px window, taken before blurring). A black piece on a dark square has
   almost the square's brightness, but its rim and highlights make the patch textured where the
   empty square is flat. Δtexture is how far the current texture falls outside the reference's
   min..max in a small neighbourhood, so a square edge nudged by a pixel doesn't count.
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
  approximate).
- `q_out` = 0.03: noise rate.
- A changed pixel inside the mask earns ln(0.55/0.03) ≈ +2.9; an unchanged pixel inside it costs
  ln(0.45/0.97) ≈ −0.77.
- `q_in` depends on what a pixel shows before and after the move. A black piece over a dark
  square (or white over light) changes far fewer pixels. How faint those pairs are is measured
  at game start from the colour distance between the learned black-piece and dark-square looks
  (and white/light): at or below `t_low` the pair uses `q_in_low` (0.25), from `t_high` up the
  normal `q_in`. On the roll-up vinyl board black-on-dark measures about 12 (faint), white-on-light 40.
- **Beyond the board edge** only the tops of tall pieces can change, but hands and sleeves rest
  there too. Off-board pixels may add to a move's score at most as much as its on-board pixels
  already give it, so a hand by the h1 corner can't make up a Kh1 when h1 itself didn't change.
  Off-board change the best move doesn't explain is reported as `off_board`: above
  `max_off_board` (3,000 px) a hand is nearby and nothing is auto-accepted. Smaller blobs there (a
  sleeve) don't count as unexplained, so they no longer block a clean move.
- **Both ends.** For each move the evidence where the piece left and where it arrived (both
  squares of the rook too when castling) is also kept separately, as the observed change rate
  relative to the expected `q_in` there. A real move changes both ends. A sleeve lying over one
  corner square changes only that one: on the real board a fake Kh1 changed g1, where the white
  king supposedly left a dark square, at 0.16–0.43 of the expected rate, while every real move's
  high-contrast ends reached at least 0.97. Faint (black-on-dark) ends are exempt, since a real
  d6-d5 can change only 0.27 of its expected rate there.
- Scores are in full-resolution pixel units, so thresholds don't depend on the work scale.

Under-promotions are scored as queen promotions (they look identical to the camera) and decided
afterwards.

The result is the ranked list, the **margin** (best minus runner-up), the **unexplained** share
(change on the board, outside the best move's mask, that forms piece-sized blobs), and the
**off-board** change (blobs beyond the board edge the best move doesn't explain). Speckle and thin
lines are removed with a morphological opening about a third of a square wide, so a vinyl board's
crease shifting doesn't count, but a second moved or bumped piece does.

Typical numbers on the real board (vinyl roll-up, camera ~75° above it):

| Read | Best score | Margin |
|---|---|---|
| Nothing moved | about −800 | under 50 |
| Clear moves (e4, Nf3, Bc4, O-O) | 4,000–19,000 | 1,800–8,000 |
| Black pawn, dark square to dark square (e7-e5, c7-c5) | 500–1,000 | 1,500–1,800 |
| Black bishop f8-e7 (dark to dark, back rank) | about 0 | about 1,000–1,400 |

## 7. Decisions and tiers — `session.py`, `confirm.py`

When the board becomes still (after following any camera shift):

1. Total change under `min_change_px` (900) → nothing happened.
2. More than 35% of the board changed → something covers the board; wait.
3. **Takeback**: taking a move back by hand changes exactly the pixels the move changed, so the
   change is also scored against the last move's own mask. If that beats every legal next move by
   `t2_margin` (and explains the change, both ends included), the move is undone; no key needed.
   This is checked even
   when a next move looks confident: putting h5 back to h7 once looked like Qxh5.
   **Adjusted piece**: the change is also scored as "only the piece on one square changed, in
   place" (its outline, slightly widened). If that beats every legal move by half `t2_margin`, and
   that piece's likeliest move would have shown up clearly at its destination but didn't, a piece
   was straightened or nudged (a pawn knocked over at the start was stood back up): the board is
   taken as it is now, no question. A faint destination (black piece arriving on a dark square)
   never counts as "didn't show up", so a faint move can't be mistaken for an adjustment.
4. No move scores at least `min_score` (500) → if one move still clearly stands out (positive
   score, margin ≥ `t2_margin`) it's re-read as in tier 2, never dropped. Otherwise, if at least
   `wait_off_board` (1,000 px) of the change lies beyond the board edge, a sleeve or hand is
   resting there: it waits ("Something is resting at the board's edge") instead of asking. If the
   change is small (under 6,000 px) it's lighting drift and is ignored; above that you're asked,
   with **X** available.
5. **Tier 1**: score ≥ 500, margin ≥ 1,500, unexplained ≤ 35%, off-board change ≤ 3,000 px, and
   every high-contrast end (expected rate ≥ `clear_end_q`, 0.45) changed at least `min_end_ratio`
   (0.75) of what was expected → accepted.
6. **Tier 2**: read 3 more still frames and average each move's score. Margin ≥ 800 (same score,
   unexplained, off-board and both-ends limits) → accepted. If the next move starts before the re-reads
   finish, it decides with the reads it has.
7. **Tier 3** (off by default): before/after photos and the top 3 moves go to a vision LLM (Anthropic
   API, `claude-sonnet-5-5`), which must answer with one of the listed moves.
8. **Tier 4**: the top 3 are drawn as numbered arrows; you press 1–3 or type it. The reference
   becomes the frame from when the question was asked, so a move made in the meantime isn't lost,
   and it's read right after you answer. That frame may still show a hand: if the next still board
   isn't a clean move from it, the board as it is then becomes the reference. While a question is
   open, a clean (tier 1) read of a later still board is accepted without an answer, and a
   takeback by hand closes the question.

Stockfish is never consulted here, so a blunder is recorded as played.

## 8. Health checks — `health.py`

- **Board window dots** (after every move, and at game start): each square is classified as
  empty / white / black from two cues learned on the starting position. **Texture** (brightness
  spread over 60% of the square) tells empty from occupied: on the real board empty squares
  measure 0.2–2.8 and every piece 5.6 or more, black on dark included. **Colour** (the centre's
  mean Lab against empty light, empty dark, white piece and black piece) tells white from black.
  Colour alone misread shaded empty squares at the board's edge as pieces. Squares whose centre is
  hidden by another piece are skipped; a dot is drawn only when the reading is at least 60% sure.
- **Sync** (every 10 moves): those readings are compared with the game. A square that confidently
  disagrees on two checks in a row is flagged in red, with Z/T/X offered.
- **Drift** (every 10 s while still, only if camera tracking couldn't follow the shift): edges of
  the top-down view are compared with the edge map from game start. If under 45% of the reference edges are still within 4 px on 3 checks in a row, the game
  pauses and asks you to recalibrate (**C**).

## 9. Promotion

When the chosen move is a promotion, the YOLO model runs on that frame. A detection counts if its
**bottom-centre** (where the piece stands, not the middle of its box) maps through `H` onto the
promotion square and its colour matches. The most confident piece type wins; otherwise a queen.

## 10. Output — `storage.py`, `engine.py`, `play.py`

- **PGN**: standard tags, the moves, and a comment per move with the time and tier. Written to a
  temporary file and swapped in, after every move. A failed write is logged and retried; it never
  stops the game.
- **Stockfish** is found from `engine.path`; if that file doesn't exist (a path written on another
  computer), from `PATH`, the usual Homebrew/Linux locations, or an `engines/` folder in the
  project. It runs in its own thread: 0.5 s per position, 3 lines, win/draw/loss when the engine
  supports it. Only the newest position is analysed. Results go to the Board window.
- **Session recording**: every frame plus the moves and calibration, for `replay`.
