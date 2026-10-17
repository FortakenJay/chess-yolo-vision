# Configuration

All settings are in [config.yaml](../config.yaml). Relative paths are relative to the project folder.
Use another file with `--config <file>`.

## camera

| Key | Default | Meaning |
|---|---|---|
| `host` | `192.168.4.1` | ESP32 address on the `ChessCam` hotspot |
| `http_port` | `80` | `/capture`, `/control`, `/status` |
| `stream_port` | `81` | `/stream` |
| `source` | `capture` | `capture` polls snapshots (works with a browser open on the camera); `stream` reads the MJPEG stream (one viewer only) |
| `framesize` | `SVGA` | Resolution by name: `VGA` 640×480, `SVGA` 800×600, `XGA` 1024×768, `HD`, `SXGA`, `UXGA`. Changing it requires recalibrating |
| `quality` | `10` | JPEG quality 4–63, lower is better and bigger |
| `vflip`, `hmirror` | `true`, `false` | Image orientation. This camera needs `vflip` |
| `extra_controls` | `{}` | Any other camera setting, e.g. `{brightness: 1, awb: 1}` |
| `apply_controls` | `true` | Push the settings above at startup |
| `connect_timeout_s` | `8` | Seconds to wait for a connection |
| `read_timeout_s` | `12` | Seconds to wait for an image |
| `reconnect_backoff_s` / `reconnect_backoff_max_s` | `1` / `10` | Wait between reconnect attempts (doubles up to the max) |

## calibration

| Key | Default | Meaning |
|---|---|---|
| `dir` | `calib` | Where the calibration is saved. Each game start finds the board again and keeps whichever fits better |
| `warp_size` | `800` | Top-down view size in px (must divide by 8) |
| `detect_attempts` | `5` | Frames tried per automatic detection |
| `max_reprojection_px` | `4.0` | Largest acceptable fit error, in top-down pixels (100 per square) |
| `canny_low`, `canny_high` | `50`, `150` | Edge detector thresholds for the drift check |

## gate — when the board counts as still

| Key | Default | Meaning |
|---|---|---|
| `pixel_threshold` | `18` | Gray-level change that counts as motion |
| `motion_fraction` | `0.004` | Share of board pixels that must change for a frame to count as motion |
| `stable_time_s` | `0.8` | Board must be still at least this long |
| `min_stable_frames` | `2` | ...and for at least this many frames |

## reader — change map and matching

| Key | Default | Meaning |
|---|---|---|
| `scale` | `0.5` | Work resolution for change maps |
| `t_low`, `t_high` | `10`, `28` | Colour distance below which a pixel is unchanged / above which it surely changed |
| `q_in` | `0.55` | Chance a pixel inside a moved piece's outline visibly changes |
| `q_out` | `0.03` | Noise rate outside |
| `q_in_low` | `0.25` | `q_in` for a piece over a square of its own tone when the board shows them faintly apart (measured at game start) |
| `texture_weight` | `5.0` | Weight of local-contrast change, which lets black pieces on dark squares register |
| `off_board_weight` | `1.0` | Weight of change beyond the board edge; never more than the move's on-board evidence |
| `height_scale` | `1.0` | Scales the assumed piece heights (taller set → raise it) |
| `min_change_px` | `900` | Less change than this: nothing happened |
| `occlusion_fraction` | `0.35` | More of the board changed than this: a hand/arm is in view, wait |
| `unexplained_alarm_px` | `6000` | Change no legal move explains: ignored below this (lighting), asks you above |

## confirm — tiers (scores in full-resolution pixels)

| Key | Default | Meaning |
|---|---|---|
| `min_score` | `500` | A move must explain at least this much change |
| `t1_margin` | `1500` | Accept immediately when the best move beats the runner-up by this much |
| `t2_margin` | `800` | Same, after averaging the re-reads |
| `max_unexplained` | `0.35` | Largest share of the change, in piece-sized blobs, the best move may leave unexplained (on the board) |
| `min_end_ratio` | `0.75` | Where a piece clearly contrasts with what's behind it, each end of a move must change at least this share of the expected rate. Stops a sleeve over one square from passing as a move |
| `clear_end_q` | `0.45` | An end counts as clearly contrasting when its expected change rate (`q_in` there) is at least this; black-on-dark ends are exempt |
| `max_off_board` | `3000` | Change beyond the board edge, in blobs the best move doesn't explain, above which a hand is assumed and nothing is auto-accepted |
| `vote_frames` | `3` | Extra still frames read in tier 2 |

Raising `t1_margin` makes it ask more often but makes mistakes less likely. Use `replay` to see
the margins of real moves before changing it.

## llm — tier 3 (off)

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Ask a vision LLM before asking you. Needs `ANTHROPIC_API_KEY` in a `.env` file in the project folder |
| `model` | `claude-sonnet-5-5` | Anthropic model |

## health

| Key | Default | Meaning |
|---|---|---|
| `sync_every_moves` | `10` | Compare the camera's view of each square with the game every N moves |
| `sync_min_confidence` | `0.9` | Only confident square readings can flag a mismatch |
| `track_min_px` | `2` | Follow the camera when the board corners shift more than this (px); see camera tracking |
| `drift_every_s` | `10` | Check whether the board or camera moved more than tracking could follow |
| `drift_threshold` | `0.45` | Share of the game-start edges that must still match |

## engine — Stockfish

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | |
| `path` | your Stockfish | Windows: the `.exe`; macOS (Homebrew): `/opt/homebrew/bin/stockfish`. If the file doesn't exist (e.g. a Windows path on a Mac), Stockfish is looked for on `PATH`, in `/opt/homebrew/bin`, `/usr/local/bin`, `/usr/games`, `/usr/bin` and in `engines/` in the project |
| `think_time_s` | `0.5` | Time per position |
| `multipv` | `3` | Lines analysed (2 are shown) |
| `show_best_moves` | `true` | `false` = training/review mode: hide best move and lines until the game ends |

## storage

| Key | Default | Meaning |
|---|---|---|
| `exports_dir` | `exports` | Where PGNs go |
| `sessions_dir` | `sessions` | Where session recordings go |
| `record_frames` | `true` | Save every frame (about 50 KB each) for replay testing |
| `white`, `black` | `White`, `Black` | Player names in the PGN |

## pieces — YOLO (promotion only)

| Key | Default | Meaning |
|---|---|---|
| `weights` | `models/pieces.pt` | 13-class piece detector |
| `conf` | `0.15` | Minimum detection confidence |
| `use_for_promotion` | `true` | `false`: always promote to a queen (fix with Z + T) |

## view, record, logging

| Key | Default | Meaning |
|---|---|---|
| `view.show_labels` | `true` | Square names on the top-down board |
| `view.captures_dir` | `captures` | Snapshots from `view` (key P) |
| `record.dir` / `record.fps` | `recordings` / `10` | Output of the `record` command |
| `logging.dir` / `logging.level` | `logs` / `INFO` | Console level; the log file always has DEBUG detail |
