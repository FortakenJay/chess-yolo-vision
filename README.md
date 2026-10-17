# Chess YOLO Vision

Reads a game played on a **real chess board** through an **ESP32 camera**, tracks every move with
python-chess, shows Stockfish's evaluation live on the computer screen, and saves the game as PGN.

```
 ESP32-WROVER CAM ──Wi-Fi "ChessCam"──▶ PC: chessreader ──▶ Board window (eval, best move, lines)
 (hotspot 192.168.4.1)                     │                ──▶ exports/game_*.pgn
                                           └─ Stockfish     ──▶ sessions/ (every frame, for replay QA)
```

- No printed markers, no special board: it finds a normal board on its own (empty or in the starting position).
- It doesn't try to recognise every piece. It compares the board before and after each move and
  picks the **legal move** that best explains what changed. That's robust to a dim, low-resolution
  camera and to pieces hiding each other.
- Unsure moves are re-checked, then shown to you as 1/2/3 choices. **Stockfish never decides which
  move was played**, so blunders are recorded as played.

---

## Contents

1. [Quick start](#quick-start)
2. [Hardware](#hardware)
3. [Playing a game](#playing-a-game)
4. [What it outputs](#what-it-outputs)
5. [How it works](#how-it-works)
6. [Project layout](#project-layout)
7. [Configuration](#configuration)
8. [Testing and QA](#testing-and-qa)
9. [Status and known limitations](#status-and-known-limitations)
10. [Further documentation](#further-documentation)

---

## Quick start

All commands are PowerShell, run from this folder (`chess-yolo-vision`).

**1. Python environment** (Python 3.12; skip if `.venv` already exists)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

**2. Stockfish.** Download Stockfish for Windows and put the path to the `.exe` in
[config.yaml](config.yaml) under `engine.path`. (Leave it empty to play without analysis.)

**3. Camera.** Flash the ESP32 once ([docs/hardware.md](docs/hardware.md)), power it, and connect this
PC's Wi-Fi to the open network **`ChessCam`**. Check it:

```powershell
.\.venv\Scripts\python run.py check
```

You should see `Camera set to SVGA 800x600` and `Stream OK`.

**4. Play.** Set up the starting position, then:

```powershell
.\.venv\Scripts\python run.py play --start
```

`--start` finds the board and starts the game automatically. Without it, press **S** once the
pieces are set up.

---

## Hardware

| Part | Notes |
|---|---|
| Freenove ESP32-WROVER CAM (OV2640, 4 MB PSRAM) | Firmware: [firmware/CameraWebServer](firmware/CameraWebServer) |
| Any normal chess board and set | Light squares and dark squares must contrast; tested on a roll-up vinyl board |
| Camera mount | Above the board, looking down. Near-overhead works best (tall pieces hide less) |
| PC | Windows 11, Python 3.12. A GPU is not needed |

The ESP32 makes its own Wi-Fi hotspot, `ChessCam`, and is always at `192.168.4.1`. Campus
networks such as eduroam can't be used (they need a username login and block devices from
talking to each other). While on `ChessCam` the PC has no internet over Wi-Fi; Ethernet still works.

Arduino IDE settings, flashing, camera orientation and the camera speed problem are all in
[docs/hardware.md](docs/hardware.md).

---

## Playing a game

1. Put the pieces in the starting position.
2. Run `.\.venv\Scripts\python run.py play --start`. Two windows open: **Camera** and **Board**.
3. Play normally. After each move, take your hand off the board; the move is read once the board has
   been still for about a second.
4. If the **Board** window shows yellow arrows numbered 1–3, press the number of the move that was
   played (or **T** to type it).
5. End with **E** (you'll be asked the result in the terminal), or **R** if someone resigned.

### Keys (click a window first)

| Key | Action |
|---|---|
| S | Start the game (when not using `--start`) |
| 1 / 2 / 3 | Pick the move when asked |
| T | Type a move in the terminal (`Nf3` or `g1f3`) |
| X | "No move happened": accept the board as it looks now (e.g. you only straightened a piece) |
| Z | Undo the last move (if the board still shows it, you'll be asked for the right one) |
| C | Recalibrate (board or camera moved). Pieces can stay where they are |
| E | End the game and enter the result |
| R | Resign (enter who resigned) |
| A | Abort the game (PGN kept, game marked unfinished) |
| N | New game after one has ended |
| Q | Quit (an unfinished game is saved with result `*`) |

More detail: [docs/usage.md](docs/usage.md).

---

## What it outputs

| Output | Where | Notes |
|---|---|---|
| **Board** window | screen | Top-down board, last move arrow, dots for what the camera sees per square, large eval + best move + win/draw/loss %, move list, Stockfish's top 2 lines, eval bar |
| **Camera** window | screen | Live camera, board outline, changed pixels in red, the outline of the move being considered, status line |
| PGN | `exports/game_<date>_<time>.pgn` | Standard PGN, rewritten after every move; each move has a comment with its time and how it was confirmed |
| Session recording | `sessions/<date>_<time>/` | Every frame (`frames/`), `frames.csv`, `moves.csv`, `calibration.json`. Used by `replay` |
| Calibration | `calib/` | Board position in the camera image, reused until the board or camera moves |
| Log | `logs/run_<date>_<time>.log` | Everything, including each move read with its score |

Training/review mode: set `engine.show_best_moves: false` to hide the best move and engine lines
during play (the eval still shows; everything is revealed when the game ends).

---

## How it works

```
frame ─▶ motion gate ─▶ change map (before vs after) ─▶ score every legal move ─▶ tiers ─▶ PGN + Stockfish
             │                                                  ▲
             └─ waits until the board is still      camera model: where each piece appears in the image
```

1. **Calibration.** OpenCV finds the board's grid (all 49 inner corners on an empty board, or the
   empty middle four ranks of the starting position) and computes a homography to an 800×800
   top-down view. The piece colours in the starting position tell which side white is on.
2. **Camera model.** From that homography the program recovers the camera's position and focal
   length, so it can predict the outline of a piece of any type standing on any square, including
   which pieces hide which.
3. **Motion gate.** Consecutive frames are compared; the board is "still" after ~0.8 s and 2 frames
   without motion. A hand over the board is motion.
4. **Change map.** The new still frame is compared with the frame from the last accepted move, after
   correcting the camera's automatic brightness changes.
5. **Move matching.** For every legal move, the pixels that move would change are predicted and
   scored against the change map (a log-likelihood ratio). Castling, en passant and captures come
   out naturally. The best move and its margin over the runner-up are kept.
6. **Tiers.** 1: clear winner → accepted. 2: re-read a few more still frames and average.
   3: (optional, off) ask a vision LLM. 4: ask you.
7. **Health checks.** Every 10 moves the per-square colours are compared with the tracked game;
   every 10 s the board edges are compared with calibration to catch a moved board or camera.
8. **Promotion.** The piece type is the only thing read by the YOLO model (`models/pieces.pt`);
   default is a queen.

Full explanation with the maths: [docs/how-it-works.md](docs/how-it-works.md).

---

## Project layout

```
chess-yolo-vision/
├── run.py                  entry point:  python run.py <command>
├── config.yaml             all settings and thresholds
├── requirements.txt
├── chessreader/            the Python package
│   ├── main.py             command line (play, replay, check, calibrate, view, record)
│   ├── play.py             live game: windows, keys, Stockfish
│   ├── session.py          one game: gating → matching → tiers → PGN (no UI; shared by play and replay)
│   ├── stream.py           ESP32 camera: settings over /control, frame reader with reconnects
│   ├── calibration.py      board detection, homography, manual corner clicking
│   ├── geometry.py         camera model, piece outlines, orientation from piece colours
│   ├── gating.py           motion gate
│   ├── reader.py           change map, per-square occupancy, YOLO wrapper
│   ├── matcher.py          legal-move scoring
│   ├── confirm.py          tiers 1–4
│   ├── health.py           sync and drift checks
│   ├── engine.py           Stockfish in a background thread
│   ├── storage.py          PGN writer
│   ├── replay.py           replay a recorded session and score it against a PGN
│   ├── display.py          drawing helpers
│   └── settings.py         config loading
├── tests/                  pytest suite (synthetic scenes) + tests/data/test_game.pgn
├── firmware/CameraWebServer/   ESP32 sketch (Arduino)
├── models/
│   ├── pieces.pt           YOLO piece detector (13 classes), used for promotions only
│   └── board_segmentation.pt   older board-finding model, kept but not used
├── docs/                   detailed documentation
└── calib/ exports/ sessions/ logs/ captures/   created at runtime (git-ignored)
```

---

## Configuration

Everything lives in [config.yaml](config.yaml), with a comment on each line. The ones you're most
likely to touch:

| Setting | Default | What it does |
|---|---|---|
| `camera.source` | `capture` | `capture` polls snapshots (works even if a browser has the camera open); `stream` uses the MJPEG stream |
| `camera.vflip` / `hmirror` | `true` / `false` | Image orientation; this camera needs `vflip` |
| `engine.path` | your Stockfish | Empty disables analysis |
| `engine.show_best_moves` | `true` | `false` = training mode |
| `confirm.t1_margin` | `1500` | How clearly a move must win to be accepted without asking |
| `storage.white` / `black` | `White` / `Black` | Player names in the PGN |
| `llm.enabled` | `false` | Tier 3; needs `ANTHROPIC_API_KEY` in a `.env` file |

Every setting is explained in [docs/configuration.md](docs/configuration.md).

---

## Testing and QA

```powershell
.\.venv\Scripts\python -m pytest tests -q
```

19 tests: calibration (corner extrapolation, never mirroring, save/load), the frame reader, move
matching on synthetic camera scenes (captures, both castles, en passant, promotion, missed moves),
and a full recorded game replayed end to end.

**Replay a real game** to measure accuracy after changing anything:

```powershell
.\.venv\Scripts\python run.py replay sessions\<folder> --pgn tests\data\test_game.pgn
```

It prints, per move, what was read, the right move, the tier and the margin, then the auto-accept
rate and the number of **wrong accepts** (target: 0). Details and current results:
[docs/testing.md](docs/testing.md).

---

## Status and known limitations

- Working: camera control, automatic calibration with pieces on the board, orientation, motion
  gating, move matching, tiers, PGN, Stockfish display, replay harness.
- **Being verified on the real board:** a full game. The first real move (e4) was read correctly;
  a too-strict confidence check made it ask instead of auto-accepting, which is fixed.
- **Camera speed:** about 0.4–2.5 frames per second with occasional stalls. The ESP32 itself is the
  bottleneck ([docs/hardware.md](docs/hardware.md#camera-speed)). It slows down move reading, it
  doesn't cause wrong moves.
- Black pieces on dark squares are hard for the per-square health check (not for move reading).
- Under-promotion relies on the YOLO model, which recognises this set poorly; default is a queen,
  and you can correct it with Z and T.

## Further documentation

| Document | Contents |
|---|---|
| [docs/hardware.md](docs/hardware.md) | ESP32 board, firmware, Arduino settings, Wi-Fi hotspot, camera orientation, camera speed |
| [docs/usage.md](docs/usage.md) | Every command and key, a typical game, recovering from mistakes |
| [docs/how-it-works.md](docs/how-it-works.md) | The pipeline in detail, with the maths |
| [docs/configuration.md](docs/configuration.md) | Every setting in `config.yaml` |
| [docs/testing.md](docs/testing.md) | Tests, the replay harness, QA results |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Symptoms and fixes |
| [docs/original-spec.md](docs/original-spec.md) | The original build specification (some parts since changed) |
