# Development

How the code is organised, the tools that keep it clean, and how a change gets verified before
it's trusted.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"     # the package + pytest, ruff, mypy, vulture, radon, pre-commit
.venv/bin/pre-commit install                     # optional: run the checks below on every commit
```

On Windows use `.\.venv\Scripts\python` (and `.\.venv\Scripts\pre-commit`). `requirements.txt` still
installs just the runtime dependencies for `run.py`.

Everything is configured in [`pyproject.toml`](../pyproject.toml): project metadata and
dependencies, ruff, mypy and pytest.

## The checks

| Tool | Command | What it guards |
|---|---|---|
| **pytest** | `.venv/bin/python -m pytest -q` | 51 tests, no camera needed (see [testing.md](testing.md)) |
| **ruff** (lint) | `.venv/bin/ruff check .` | pycodestyle, pyflakes, import order, bugbear, pyupgrade, simplify, pylint, naming, and a complexity cap of 15 per function |
| **ruff** (format) | `.venv/bin/ruff format .` | One formatting style, 120 columns |
| **mypy** | `.venv/bin/mypy` | Types across `chessreader/`, with the pydantic plugin and `check_untyped_defs` |
| **vulture** | `.venv/bin/vulture chessreader tests tools tools/vulture_whitelist.py --min-confidence 60` | Dead code. Names used only by a framework are listed in `tools/vulture_whitelist.py` |
| **radon** | `.venv/bin/radon cc chessreader -s -n C` | Complexity report, for finding the next function to split |
| **pre-commit** | `.venv/bin/pre-commit run --all-files` | ruff, ruff format, mypy and pytest together, using the venv's own tools (nothing is downloaded) |

All of them pass on `main`. A few ruff rules are switched off on purpose, each with its reason in
`pyproject.toml`: magic numbers (vision thresholds are named in `config.yaml` or explained where
used), many-argument functions (drawing helpers mirror OpenCV), imports inside functions (YOLO and
the Anthropic SDK stay off the startup path), and `int(round(x))`, which stays explicit because
OpenCV needs real Python ints from numpy values.

## Verifying a change to the reader

Unit tests use synthetic scenes. The recorded games are the real test:

```bash
.venv/bin/python tools/eval_sessions.py              # every game with a truth timeline
.venv/bin/python tools/eval_sessions.py 20261017_104913 -v
```

`tools/eval_sessions.py` plays each recorded session through the live code like a player who only
answers a question when the right move is offered, and checks the program's position on every still
frame against `tools/session_timelines.json` (what was really on the board, frame range by frame
range; takebacks included). The target is **0 wrong frames**; it exits 1 otherwise. Sessions live in
`sessions/` (git-ignored), so this runs where they were recorded or copied.

For a refactor that must not change behaviour, compare before and after:

```bash
for s in $(ls sessions); do .venv/bin/python run.py replay sessions/$s > /tmp/after_$s.txt; done
```

and diff against the same run on the old code. The 2026-10-21 cleanup was checked this way: all 12
sessions replayed byte-identically, and `eval_sessions.py` gave the same 0-wrong results before and
after.

## Code structure

```
run.py → main.py (commands) ─┬─ play.py      LiveGame: camera loop, two windows, one handler per key
                             ├─ replay.py    a recorded session or video through the same pipeline
                             └─ calibration.py, stream.py (check / calibrate / view / record)

GameSession (session.py) — one game, no UI; play.py and replay.py drive it frame by frame
  ├─ Pipeline         built from the calibration: BoardCamera, MoveMatcher, ChangeReader, MotionGate
  ├─ process(frame)   → a handler per phase: _on_playing, _on_voting, _on_question, _on_waiting_for_llm
  ├─ _decide(result)  nothing changed → covered → takeback → adjusted piece → weak move → tier 1 / vote
  ├─ GameRecord       the PGN (storage.py), with ECO/Opening tags from openings.book_for(settings)
  └─ SessionRecorder  frames.csv / moves.csv / calibration.json for replay (storage.py); one MoveLog per row
```

- **`GameSession`** holds the game logic and nothing else. Its camera-dependent parts live in one
  `Pipeline`, built when a game starts and rebuilt when the calibration changes (camera tracking,
  **C**). `session.ready` says whether it exists; `session.cam`, `.matcher`, `.reader` and `.gate`
  raise a clear error if used before a game starts, instead of failing on `None` somewhere deep.
- **`LiveGame`** (play.py) is the only place with windows and keys. Keys map to small methods in a
  table, so a new key is one entry and one method. `tests/test_play.py` drives it with a fake camera
  and scripted key presses.
- **Settings** are pydantic models (`settings.py`), loaded from `config.yaml`; every key is documented
  in [configuration.md](configuration.md).
- **Shared loaders, one copy each:** `openings.book_for(settings)` (cached; a broken book warns once
  and switches openings off), `Calibration.load_saved(dir)`, `replay.frames_from(path)` (lazily,
  also used by `tools/eval_sessions.py`), and the session file names on `SessionRecorder`.
- **Threads:** Stockfish and the optional LLM run in background threads. `LiveGame` registers one
  engine listener that always feeds the current session, so resets don't pile up listeners.
- **Not done yet:** the state that exists only during a game (pipeline, record, drift monitor,
  reference frame) is still spread over `GameSession` attributes; grouping it into one `Game` object
  is the next structural step. Some tuning numbers (e.g. `contrast < 25` at game start) are still
  literals rather than settings.

## Conventions

- Type hints everywhere; `from __future__ import annotations` at the top of modules.
- Docstrings say *why*, in plain words, and quote the real-board numbers a threshold came from.
- A fix for something seen live comes with a test that fails on the code before it (see the
  [series log](series-log.md)), and a replay of the session that showed it.
- Commit messages: what changed and why, with the replay/eval numbers that back it.
