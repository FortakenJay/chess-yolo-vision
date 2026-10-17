> **Historical document.** This is the specification the project was rebuilt from. It no longer
> matches the code in these ways:
> - **Supabase and the e-ink display were removed.** Games are saved as PGN in `exports/`; the eval
>   and best move are shown in the Board window on the computer screen.
> - **Moves are read from before/after change, not from YOLO piece detection**, because the YOLO model
>   found only 9–15 of 32 pieces on this camera. YOLO is used for promotions only.
> - The camera is polled via `/capture` by default; the board can be calibrated with the pieces in
>   the starting position (not only empty); tier 3 (LLM) is off by default.
> - The project is no longer under `redo/`; it is the repository root, with the package in `chessreader/`.
>
> See the [README](../README.md) and the other files in `docs/` for how it works now.

You are helping me build a physical chess board reader in Python. A Freenove ESP32-WROVER CAM streams video of a real chess board to my PC. The PC reads the board, tracks the game, analyzes it with Stockfish, and saves it. Build it as a clean, modular project I can run and test step by step.

## Context

### Camera and network (already working, do not change the firmware unless I ask)
- Board: Freenove ESP32-WROVER CAM (classic ESP32 + OV2640, 4 MB PSRAM). Firmware is Espressif's CameraWebServer example in `redo/CameraWebServer/`, pins = `CAMERA_MODEL_WROVER_KIT`. Arduino settings: board "ESP32 Wrover Module", PSRAM Enabled, Huge APP partition. The serial log prints `[alive] ... free psram N` once a second; N must be > 0.
- The ESP32 runs in **softAP mode**: it creates an open Wi-Fi network `ChessCam` and is always at **192.168.4.1** (no DHCP reservation, no eduroam; eduroam is WPA2-Enterprise with client isolation and can't be used).
- Endpoints:
  - MJPEG stream: `http://192.168.4.1:81/stream`
  - Single JPEG: `http://192.168.4.1/capture` (port 80, works while the stream is open)
  - Settings: `http://192.168.4.1/control?var=<name>&val=<n>` (e.g. `framesize`, `quality`, `vflip`, `hmirror`, `brightness`)
- The firmware boots at **QVGA (320×240)**. On startup, the Python side must set `framesize` (default SVGA 800×600; configurable) and `quality` through `/control` before reading the stream. The numeric `framesize` values differ between esp32-camera library versions. Don't hard-code a number: after setting it, read `GET http://192.168.4.1/status` (JSON with the current `framesize`) and check the first decoded frame's actual size.
- The stream server handles **one viewer at a time**. Python must be the only stream consumer (no browser tab open on `:81/stream`).
- Keep all of the above as config values (base URL, ports, framesize, quality, flips), not hard-coded.
- **The PC is on two networks at once:** Wi-Fi → `ChessCam` (no internet), Ethernet → campus network (internet). Camera and e-ink traffic go over `ChessCam`; Supabase and the Anthropic API go over Ethernet. If there's no internet, everything cloud-related must degrade gracefully: queue Supabase writes locally, skip Tier 3 and go straight to Tier 4, and never block or crash the vision loop.

### Piece model (already trained)
- Weights: `models/pieces.pt` (repo root). Ultralytics **YOLO26m** detector (NMS-free, needs `ultralytics>=8.4.0`), trained at `imgsz=800`.
- It has **13 classes**, not 12:
  `{0: 'bishop', 1: 'black-bishop', 2: 'black-king', 3: 'black-knight', 4: 'black-pawn', 5: 'black-queen', 6: 'black-rook', 7: 'white-bishop', 8: 'white-king', 9: 'white-knight', 10: 'white-pawn', 11: 'white-queen', 12: 'white-rook'}`
- Class 0 `bishop` has no color (a Roboflow leftover). Treat it as "bishop, color unknown": it counts as occupied but contributes no color evidence. Don't alias it to white the way the old code does.
- Read the class names from `model.names` at load time; don't hard-code the index order.
- The old pipeline ran this model on the 800×800 warped board. In step 4, check whether it works better on the warp or on the raw frame (then map boxes through the homography), and tell me which.

### Existing code (reuse, don't break)
- Repo root: `chess-yolo-vision/` on **Windows 11**, PowerShell, Python **3.12** venv in `.venv/`, CUDA torch (cu121), RTX 4070 Laptop GPU. Secrets live in `.env` (`SUPABASE_URL`, `SUPABASE_KEY`; add `ANTHROPIC_API_KEY`) and load with python-dotenv.
- The old version lives in `src/` + `main.py` + `tests/` (phone-camera stream, YOLO corner model, FEN smoothing). **Build the new version under `redo/`** as its own package and leave `src/` alone. Port or copy anything useful from it, in particular:
  - `src/settings.py` (pydantic config loading)
  - `src/pgn_exporter.py`
  - `src/supabase_client.py` (insert, plus fallback JSON to `exports/failures/` on failure)
  - `src/display.py` (OpenCV overlays)
- `models/corners.pt` doesn't exist and isn't needed: calibration replaces it.
- Keep the old keyboard controls in the debug window: `s` start, `e` end (prompt for result), `r` resign, `a` abort (no save), `q` quit. Add `c` to recalibrate.

### Supabase
- The existing table is `games`, which the old code inserts into with these columns: `played_at` (ISO timestamp), `result` (`white`/`black`/`draw`/`unknown`), `opening_name`, `opening_eco`, `pgn`, `starting_fen`, `total_moves`, `capture_device`, `notes`; it returns `id`.
- Set `capture_device` to `"esp32-wrover-cam"`.
- Incremental writes need an in-progress row (insert at game start, update `pgn`/`total_moves` per move). Propose any new columns, such as a status field, and **ask me before changing the schema**.
- My website (chess.shelfpos.net) runs a move classifier on stored games, so the PGN must be standard and complete.

### E-ink display
- A second ESP32 drives an e-ink display showing the eval bar and best move. It joins the `ChessCam` network as a client, so the PC reaches it at a `192.168.4.x` address (the PC itself gets one too, usually `.2`).
- It accepts JSON over HTTP POST at <EINK_URL>. Ask me for it; suggest giving the e-ink ESP32 a static IP such as `192.168.4.50`.
- Make the transport swappable for MQTT later.

### Other constraints
- I have no printer, so **no ArUco or printed markers** (ignore `aruco_markers/` and `code/generate_aruco_markers.py`).
- Stockfish path: <STOCKFISH_PATH>, a Windows `.exe`. Ask me.
- Anthropic model: config value, default `claude-sonnet-5-5`; key from `ANTHROPIC_API_KEY`.

## Pipeline to implement
1. **Stream intake:** on startup, push the `/control` settings. Then a background thread reads `cv2.VideoCapture(url)` nonstop and keeps only the newest frame (no stale buffered frames). Handle reconnects if the stream drops, and log clearly when the PC isn't on `ChessCam` (192.168.4.1 unreachable).
2. **Calibration** (once per session, on an EMPTY board):
   - Use `cv2.findChessboardCornersSB(gray, (7,7))` to get the 49 inner corners, then extrapolate one square outward to get the 4 outer corners.
   - Fallback: show a frame and let me click the 4 outer corners with `cv2.setMouseCallback`.
   - Compute the homography to an 800×800 top-down view (100 px per square).
   - Save the homography, a warped empty-board reference image, and its Canny edge map to disk (JSON plus images) so I can reload them.
   - After I set up the pieces, detect orientation from the starting position (white on ranks 1–2) and fix the square mapping. This replaces the old 4-option orientation menu.
3. **Motion gating:** diff consecutive warped frames. Only read the board once it has been stable for a configurable time (~0.5–1 s, N frames below a threshold) AND it differs from the last accepted position. Ignore frames with a hand over the board.
4. **Square reading:** for all 64 squares, output occupancy/color probabilities (empty/white/black) plus a piece type with confidence from YOLO. Assign each YOLO box to a square by its BOTTOM-CENTER (piece base), not the box center. Structure the code so I can swap in a 13-class square-crop classifier later.
5. **Legal-move matching (the core):** keep a python-chess `Board` from the starting position. On each new stable read, score every legal move by the log-likelihood of its resulting occupancy/color map against the observed probabilities. Return the best move and its margin over the runner-up. Promotion is the only case that uses piece type: check YOLO on the promotion square and default to queen.
6. **Tiered confirmation:**
   - Tier 1: margin above threshold → accept automatically.
   - Tier 2: low margin → re-read 5–10 stable frames and vote.
   - Tier 3: still ambiguous, or no legal move fits → send the warped image plus the top 3 candidate moves to a vision LLM (Anthropic API). Prompt: "Which of these moves was played? Reply with exactly one UCI move from the list." Reject any answer not in the list. Skip this tier when offline.
   - Tier 4: still unsure → ask me in the terminal for v1 (a simple local web UI later) to pick the move.
   - NEVER use Stockfish to decide which move was played (it would hide real blunders).
7. **Health checks:**
   - Every ~10 moves, compare a full YOLO board read against the tracked FEN. On a persistent mismatch, flag it and offer a resync.
   - Periodically compare the warped board border against the calibration edge map. If it drifts beyond a threshold, pause and prompt me to recalibrate by clicking 4 corners.
8. **Stockfish:** python-chess `SimpleEngine` (path from config). Analyse with `Limit(time=0.5)`, `multipv=3`, run asynchronously so it never blocks the vision loop. Output eval (cp or mate), best move, PV, and WDL. Include a "training/review mode" flag that controls whether best moves get shown live.
9. **E-ink output:** send `{eval, best_move, pv, fen}` as JSON to the e-ink ESP32 after each analysis. Use a short timeout and don't block on failures.
10. **Storage:** append each accepted move to a PGN in `exports/` (with headers and timestamps). Write the game to Supabase incrementally and at game end, using the `games` table above. On failure, save the payload to `exports/failures/` and retry when the connection is back.

## Project requirements
- Python 3.12, OpenCV, ultralytics>=8.4.0, python-chess, numpy, requests, supabase, anthropic, pydantic, PyYAML, python-dotenv, pytest.
- Modules (under `redo/`): `stream.py`, `calibration.py`, `gating.py`, `reader.py`, `matcher.py`, `confirm.py`, `health.py`, `engine.py`, `eink.py`, `storage.py`, `main.py`, `config.yaml`.
- Unit tests in `redo/tests/` for at least `matcher.py` (synthetic probability maps, including castling, en passant, and promotion), `gating.py`, and calibration corner extrapolation.
- A debug view (OpenCV window) that shows the live warped board with the grid overlay, per-square occupancy, the last accepted move, and the current confidence tier.
- A replay/test mode that runs the pipeline on recorded video files instead of the live stream. Per move, it logs the tier used, the margin, and whether it matched a ground-truth PGN if I provide one. Report the auto-accept rate and the wrong-accept count. Target: zero wrong accepts. Include a small recorder script that saves the live stream to a video file so I can build test sets.
- Log everything useful for debugging; keep thresholds in `config.yaml`.
- Give run commands as PowerShell using `.\.venv\Scripts\python`.

## How to work
Build in this order, and stop after each step so I can test it on my hardware before you continue:
1. Stream (including `/control` setup) plus calibration plus warp (confirm a clean top-down view)
2. Motion gating
3. Occupancy reading plus legal-move matching
4. YOLO piece types plus promotion handling (including the warp-vs-raw check above)
5. Tier 2–4 confirmation plus health checks
6. Stockfish plus e-ink
7. Supabase storage plus the replay/test harness

Ask me before guessing on anything marked <...> above (EINK_URL, STOCKFISH_PATH) and before any Supabase schema change. Start with step 1.
