# Chess Vision Recorder

Real-time chess game recorder using a phone camera stream (Expo + desktop bridge), Ultralytics YOLO26 detection (NMS-free end-to-end), perspective warping, move legality checks, PGN export, and optional Supabase persistence.

This repository is designed for:
- **Live board capture** from iPhone/Android camera via Expo + bridge MJPEG stream
- **Board understanding** (corner detection + piece detection)
- **Game reconstruction** (FEN smoothing + legal move parsing with `python-chess`)
- **Export + storage** (PGN files locally, optional Supabase insert)

---

## 1) Current Project Structure

```text
Chess-YOKOv8-vision/
├── code/
│   ├── esp32_cam_stream.ino      # ESP32-CAM firmware (MJPEG /stream)
│   ├── train_models.py           # trains pieces + corners models
│   └── view_esp32_stream.py      # quick stream viewer
├── datasets/
│   ├── Chess_Pieces.yolov8/
│   └── Chessboard_detection-4Corners.yolov8/
├── exports/                      # PGN output and failure archives
├── models/                       # final model weights copied here
├── src/
│   ├── stream.py
│   ├── corners.py
│   ├── warp.py
│   ├── pieces.py
│   ├── fen.py
│   ├── validator.py
│   ├── smoother.py
│   ├── move_recorder.py
│   ├── pgn_exporter.py
│   ├── supabase_client.py
│   ├── display.py
│   ├── pipeline.py
│   ├── settings.py
│   ├── errors.py
│   └── types.py
├── tests/
├── config.yaml
├── .env.example
├── requirements.txt
└── main.py
```

---

## 2) How It Works (High-Level)

Per frame:
1. Read frame from MJPEG stream (phone bridge by default, ESP32 optional).
2. Optionally rotate frame 180 in software (`config.yaml`).
3. Detect board corners (YOLO26s corners model, filtered to the `corner` class).
4. Warp to top-down board view.
5. Detect pieces (YOLO26m pieces model) and map to board squares.
6. Build FEN from mapped board.
7. Validate position legality.
8. Smooth FEN over a temporal window.
9. If stable FEN changed, parse legal move and append to game.
10. Render two OpenCV windows: raw feed + warped board view.

Game end:
1. Export PGN to `exports/`.
2. Attempt Supabase insert.
3. If Supabase fails, save fallback payload in `exports/failures/`.

---

## 3) Requirements

- Windows 10/11 (or Linux/macOS)
- Python **3.12** recommended in this repo currently
- NVIDIA GPU supported (tested with RTX 4070 Laptop GPU)
- Phone with Expo Go (recommended camera source)
- ESP32-CAM is optional legacy hardware source

Python dependencies are in `requirements.txt`.

---

## 4) Environment Setup

From repository root (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_publishable_or_service_key
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

---

## 5) Phone Camera Setup (Default: Native iPhone Streamer + Bridge)

Use this by default (no ESP32 required).

### Preferred (no install): Safari web streamer (Next.js)

Use `web_streamer/` on iPhone Safari — live camera via browser, no AltStore or Expo:

```powershell
cd web_streamer
npm install
npm run dev
```

Open `http://<PC_LAN_IP>:3000` on iPhone. See `web_streamer/README.md`.

### Native iPhone streamer (AltStore path)

Use the native app in `ios_phone_streamer/` for true video-frame streaming:

1. On a Mac, generate/open the iOS project:

   ```bash
   brew install xcodegen
   cd ios_phone_streamer
   xcodegen generate
   open PhoneStreamer.xcodeproj
   ```

2. Build and sideload with AltStore.
3. Continue with steps below to run the desktop bridge + pipeline.

For native app setup details, see `ios_phone_streamer/README.md`.

### Fallback: Expo app (snapshot-based)

1. Start the phone bridge on your PC (repo root):

   ```powershell
   .\.venv\Scripts\python code/phone_stream_bridge.py --host 0.0.0.0 --port 8080
   ```

2. Start the Expo mobile app:

   ```powershell
   cd mobile
   npm install
   npx expo start
   ```

3. Open Expo Go on iPhone and scan the QR code.
4. In the app, set:
   - Host = your PC LAN IP (run `ipconfig` on Windows)
   - Port = `8080`
5. Press **Start Streaming** in the app.
6. Verify stream on desktop:

   ```powershell
   .\.venv\Scripts\python code/view_esp32_stream.py --url http://127.0.0.1:8080/stream
   ```

7. Point the pipeline to the phone bridge in `config.yaml`:
   - `esp32_url: "http://127.0.0.1:8080/stream"`

8. Run the main app:

   ```powershell
   .\.venv\Scripts\python main.py
   ```

Notes:
- Keep the iPhone app in the foreground while streaming.
- Allow inbound TCP port `8080` on Windows firewall (Private network) if connection fails.
- For Expo details, see `mobile/README.md`.
- For native iOS details, see `ios_phone_streamer/README.md`.

---

## 6) ESP32-CAM Firmware Setup (Optional Legacy Source)

File: `code/esp32_cam_stream.ino`

1. Open in Arduino IDE.
2. Set:
   - `WIFI_SSID`
   - `WIFI_PASSWORD`
3. Set board to **AI Thinker ESP32-CAM**.
4. Upload.
5. Open Serial Monitor and read stream URL:
   - `http://<esp32-ip>:81/stream`

## 7) Config File

File: `config.yaml`

Important keys:
- `esp32_url`: MJPEG endpoint
- `corner_model_path`: `models/corners.pt`
- `piece_model_path`: `models/pieces.pt`
- `corner_conf`, `piece_conf`: detection confidence thresholds
- `smoothing_frames`: temporal FEN window
- `warp_size`: board warp output size (divisible by 8)
- `rotate_180`: software 180-degree rotation fallback
- display toggles:
  - `show_square_labels`
  - `show_confidence_scores`
  - `show_piece_count_hud`
  - `highlight_last_move`

---

## 8) Training Models

File: `code/train_models.py`

### Dataset sources used

- Corners dataset: [Roboflow Universe - Chessboard detection (4 corners)](https://universe.roboflow.com/chessboarddetection/chessboard-detection-4-corners)
- Pieces dataset: [Roboflow Project - Chess Full (jay-tmdyh)](https://app.roboflow.com/jay-tmdyh/chess-full-cnkvv/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

What the script does:
- Detects dataset yaml recursively under `datasets/**/data.yaml`
- Matches naming variants (spaces/underscores/case-insensitive)
- Auto-fixes Roboflow path issues:
  - creates `data.autofix.yaml` with absolute paths
  - falls back `val -> train` when val set is missing
- Trains both:
  - pieces model (base: `yolo26m.pt`)
  - corners model (base: `yolo26s.pt`)
- Weights are auto-downloaded by Ultralytics on first run (requires `ultralytics>=8.4.0`).
- `yolo26m` at `imgsz=800, batch=8` is tight on 8 GB VRAM. If pieces training OOMs, lower `batch` to 4 in `code/train_models.py`.
- Copies best weights to:
  - `models/pieces.pt`
  - `models/corners.pt`

Run:

```powershell
.\.venv\Scripts\python code/train_models.py
```

### Current memory-safe training defaults

To avoid Windows pagefile/multiprocessing crashes, script uses:
- `workers=0`
- `cache=False`
- `batch=8`
- `imgsz=800`

These are intentional stability defaults for your environment.

### If only `pieces.pt` exists and `corners.pt` is missing

This means pieces training completed but corners training did not finish/copy successfully.
In that case, run corners-only training:

```powershell
.\.venv\Scripts\python -c "import runpy; ns=runpy.run_path('code/train_models.py'); ns['train_corners'](device=0)"
```

Then verify:

```powershell
ls models
```

You should have both `models/pieces.pt` and `models/corners.pt` before running `main.py`.

---

## 9) Running the App

```powershell
.\.venv\Scripts\python main.py
```

Optional auto-start game:

```powershell
.\.venv\Scripts\python main.py --start
```

Startup prompt:
- `Is white playing from bottom? (y/n)`

OpenCV windows:
- `Chess Vision` (raw stream + HUD/corners)
- `Board View` (warped board + overlays)

Keyboard controls:
- `s` start game
- `e` end game (prompts result)
- `r` resign (prompts side)
- `a` abort game (no save)
- `q` quit

---

## 10) Supabase

Module: `src/supabase_client.py`

Behavior:
- Validates payload before insert
- Inserts into `games` table
- If insert fails:
  - writes fallback JSON payload to `exports/failures/`
  - raises `SupabaseInsertError`

Gameplay is designed to keep local PGN as source-of-truth.

---

## 11) Core Modules and Responsibilities

- `src/stream.py`  
  ESP32 URL validation, reachability check, frame validation, retry/backoff.

- `src/corners.py`  
  Corner model inference, confidence filtering, sorting, convex/area/bounds checks.

- `src/warp.py`  
  Perspective transform validation and top-down board generation.

- `src/pieces.py`  
  Piece detection, label normalization, board mapping, plausibility checks.

- `src/fen.py`  
  Converts board map to complete FEN with strict rank/key/value validation.

- `src/validator.py`  
  Legal position checks via `python-chess`.

- `src/smoother.py`  
  Temporal consensus over recent FENs.

- `src/move_recorder.py`  
  Legal move matching from stable FEN changes, SAN/UCI tracking.

- `src/pgn_exporter.py`  
  Builds and writes PGN, validates output file.

- `src/display.py`  
  All rendering logic (HUD, boxes, labels, highlights, board grid).

- `src/pipeline.py`  
  End-to-end orchestration and runtime state machine.

---

## 12) Tests

Run tests:

```powershell
.\.venv\Scripts\python -m pytest -q
```

Current suite covers:
- stream validation
- corner logic
- warp checks
- piece mapping/plausibility
- FEN generation
- legality validation
- smoothing behavior
- move recorder scenarios
- PGN exporter
- Supabase fallback
- display output shape/behavior

---

## 13) Troubleshooting

### A) `CUDA is not available`
- Ensure CUDA wheel installed in `.venv`:
  - `torch==...+cu121`
- Re-check with `torch.cuda.is_available()`

### B) `Torch not compiled with CUDA enabled`
- You installed CPU-only torch.
- Reinstall torch from CUDA index URL.

### C) `WinError 1455 ... cublas64_12.dll`
- Windows pagefile too small during worker spawn.
- Current script mitigates with `workers=0` and `cache=False`.
- Also keep machine awake during training.

### D) `images not found` from Ultralytics
- Dataset path mismatch in Roboflow export.
- Script now auto-fixes yaml paths and folder naming variants.

### E) Repeated `corrupt JPEG restored and saved`
- Common with Roboflow-exported images.
- Ultralytics is repairing them; warning is usually non-fatal.

### F) Training seems stuck
- Wait for first epoch logs after dataset scan.
- Watch GPU usage in Task Manager.
- Ensure terminal is still producing output.

---

## 14) Operational Notes

- Keep `.venv/` out of git (already ignored).
- Keep `.env` private.
- Ensure `models/corners.pt` and `models/pieces.pt` exist before `main.py`.
- For best move quality, use stable camera mount and consistent board lighting.

---

## 15) Quick Start (Minimal)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
python -m pip install -r requirements.txt
copy .env.example .env
python code/train_models.py
python main.py
```

