# Troubleshooting

The log in `logs/` has the details of every run, including each move read with its score, margin
and unexplained share.

## Camera

| Symptom | Cause / fix |
|---|---|
| `Can't reach ESP32 at 192.168.4.1 ... Is Wi-Fi connected to ChessCam?` | Connect the PC's Wi-Fi to `ChessCam`. After the ESP32 restarts, Windows sometimes joins another network |
| `PC is on the camera network but the ESP32 isn't answering` | The ESP32 is stuck. Press its reset button, then check the Serial Monitor |
| Stream fails with "connection reset" | Another viewer holds the single stream slot (often a browser tab on `192.168.4.1`). Close it, or use `camera.source: capture` (default) |
| Very slow frames, timeouts, pings of hundreds of ms | The ESP32 itself is slow: see [hardware.md](hardware.md#camera-speed). Use a good 5 V supply, flash the current firmware, keep the antenna end uncovered and 1–3 m from the PC |
| Serial Monitor: `Camera init failed` | Wrong board in the Tools menu, or a loose camera ribbon. Use **ESP32 Wrover Module** |
| Serial Monitor: `free psram 0` / `WARNING: no PSRAM found` | Tools → PSRAM → **Enabled**, upload again |
| Windows says the hotspot password is wrong | The current firmware has no password. If you add one: Windows first asks for a router PIN; choose "Connect using a security key instead" |
| Image upside down or mirrored | `camera.vflip` / `camera.hmirror` in `config.yaml`. Printed coordinates on the board must read normally |
| `Asked for SVGA ... but camera reports framesize=...` | The firmware's camera library numbers sizes differently; tell the program the frame size you get, or reflash with the current Arduino esp32 package |

## Board finding

| Symptom | Fix |
|---|---|
| `Board not found` | The whole board must be in view, evenly lit, either empty or in the starting position (the middle four ranks must be empty). Or press **M** in the calibration window and click the 4 corners |
| `fit error ... too high` | Lens distortion or a very bent board. Flatten the board or click the corners |
| `Can't see the pieces clearly (contrast ...)` when starting | The starting position isn't set up, or the light is too dim to tell white from black. The log line `Orientation contrast: board found now N, saved calibration M` shows both fits; ~85 is normal |
| Grid lines don't sit on the squares (`view`) | Recalibrate (`calibrate` or **C**). In a game this is fixed at the next start, which finds the board again |
| `The board or camera moved` | Small shifts are followed automatically (`Camera moved ... followed it` in the log). This message means tracking couldn't follow it (a big jump, or the board covered): press **C**; pieces can stay. Fix the camera mount so it can't swing |
| `Frame is ... but calibration was made at ...` | `camera.framesize` changed; recalibrate |

## Reading moves

| Symptom | Fix |
|---|---|
| A move takes a while to appear | It waits for 2 still frames; at ~1 frame per second that's 2–3 s. Keep hands off the board until it appears |
| It asks too often | Check the margins of real moves in the log or with `replay`, then lower `confirm.t1_margin` carefully. Replay to make sure wrong accepts stay at 0 |
| `The board changed but no legal move fits` | A piece was bumped, a move was made that python-chess thinks is illegal, or two moves were made without pausing. Fix the board and press **X**, or **T** to type the move |
| `Camera disagrees on ...` | Look at the red squares. **Z**/**T** if the game is wrong, **X** if the board is right |
| `Something is covering the board` | A hand or arm was still over the board when it went still. Move it away |
| A move was recorded wrongly | **Z**, then the right number or **T** |
| You took a move back by hand | It's undone automatically (`was taken back` in the log). Put the piece back on its square and keep hands off until the board is still |
| It asks while your hand rests next to the board | Expected: change beyond the board edge above `confirm.max_off_board` means a hand is there, so nothing is accepted automatically. Move the hand away; a clear re-read then answers the question by itself |
| A black move between two dark squares needs confirming | It's faint to this camera. The log shows `black pieces on dark squares: contrast N, q_in Q` at game start. Better, more even light helps most; raising `reader.texture_weight` a little (default 5) can too. Check with `replay` that no extra questions appear |
| A move appeared that nobody made | Usually a sleeve or hand lying over a square near the edge. Both ends of a move must now change, so this should come up as a question instead; if it still happens, keep the session folder: replay reproduces it exactly |
| Dots on empty squares, or missing on pieces | Dots are refreshed after every move and drawn only when the reading is at least 60% sure. Something resting on a square (a sleeve, a shadow) really does read as occupied. If dots are wrong on a clear board, start a new game (**N**) so the looks are learned again in the current light |
| It asks after you only straightened a piece | Straightening is recognised on its own when the piece's move would have shown clearly where it went (`adjusted in place` in the log). If it still asks, press **X** |
| "Something is resting at the board's edge; waiting." | No move fits and the change runs off the board: move the sleeve or hand away |
| Promotion recorded as a queen but it was a knight | **Z**, then **T** `e8=N` (for example) |
| Keys do nothing | Click one of the program's windows first |
| The game got messed up | **R** resets: the game so far is kept as an unfinished PGN, a new one starts from the starting position |
| The opening name is missing or wrong | The log says whether the book loaded. Names come from `chessreader/data/openings.tsv`; fix or add a line there (`eco`, `name`, `pgn`) |

## Engine

| Symptom | Fix |
|---|---|
| `Stockfish not found ...; analysis off` (eval bar never moves, no best move) | Stockfish isn't installed on this computer, or `engine.path` points somewhere else. Install it (macOS: `brew install stockfish`; or download it from stockfishchess.org and put it in `engines/` in the project) or set `engine.path`. A path that doesn't exist falls back to one found on `PATH` |
| No eval shown | Stockfish needs ~0.5 s after each move; check the log for `Stockfish ready` and for Stockfish errors |
