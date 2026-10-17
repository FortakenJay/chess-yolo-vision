# Usage

Run everything from the project folder with the project's Python:

```powershell
.\.venv\Scripts\python run.py <command> [options]
```

```bash
.venv/bin/python run.py <command> [options]     # macOS / Linux
```

The examples below use the Windows form.

## Commands

| Command | What it does |
|---|---|
| `play` | Play a game (default). Opens the Camera and Board windows |
| `play --start` | Same, but finds the board and starts the game automatically when the starting position is seen |
| `replay <folder or video> [--pgn file] [--llm]` | Run a recorded game through the reader and report accuracy |
| `check` | Checks the Wi-Fi/camera, applies camera settings, measures frames per second. No windows |
| `calibrate` | Find the board again (interactive), then show the live views |
| `view` | Live camera and top-down board with the grid; for checking calibration |
| `record` | Save the live camera to an `.mp4` in `recordings/` |

Every command accepts `--config <file>` to use a different settings file.

## A typical game

1. Connect the PC's Wi-Fi to `ChessCam` and run `check` if you're not sure the camera is up.
2. Set up the starting position.
3. `.\.venv\Scripts\python run.py play --start`
4. The Board window shows the board from above with white at the bottom, and the eval panel.
5. Make a move, take your hand away (also from the board's edges). Within a couple of seconds the
   move appears as an arrow, in the move list, and Stockfish's eval updates.
6. If it asks (yellow arrows numbered 1–3), press the right number. If it asked because your hand
   was still near the board, moving the hand away is often enough: a clear re-read is accepted.
7. Finish with **E** and type the result (`w`, `b`, `d` or `u`) in the terminal, or **R** for a
   resignation. Then **N** for a new game, or **Q** to quit.

The PGN is in `exports/` from the first move on; it's rewritten after every move, so it's never lost.

## Keys

Click one of the windows first so it receives the key presses.

| Key | When | Action |
|---|---|---|
| S | before the game | Start (looks for the starting position; says why if it can't) |
| 1, 2, 3 | when asked | Choose that move |
| T | any time in a game | Type a move in the terminal: SAN (`Nf3`, `O-O`, `exd6`, `e8=Q`) or UCI (`g1f3`) |
| X | any time | "Nothing was moved": take the board as it looks now as the current position |
| Z | any time | Undo the last move. If the board still shows a move, you're asked which one it was |
| C | any time | Recalibrate. Only needed if the board or camera moved more than it can follow. In a game, the pieces stay; orientation is recovered from the current position |
| E | in a game | End and enter the result |
| R | in a game | Resignation: enter who resigned |
| A | in a game | Abort. The PGN is kept with result `*` |
| N | after a game | Start a new game (same calibration) |
| Q | any time | Quit. An unfinished game is saved with result `*` |

## Fixing mistakes

| Situation | Do this |
|---|---|
| It recorded the wrong move | **Z**, then pick or type (**T**) the right one |
| You took a move back on the board | Nothing: it's undone automatically once the board is still. **Z** also works |
| You straightened a piece | Usually nothing: it's recognised as an adjustment. If it asks anyway, **X** |
| You knocked pieces over and put them back | **X** once the board is right |
| "The board or camera moved" | Small shifts are followed on their own; this means it couldn't. **C** to recalibrate. Pieces can stay |
| "Camera disagrees on e4, ..." | Check those squares. **Z**/**T** if the game is wrong, **X** if the board was only touched |
| A pawn promoted to something other than a queen | The camera tries to read the piece; if it's wrong, **Z** then **T** with e.g. `e8=N` |

## Training / review mode

Set `engine.show_best_moves: false` in `config.yaml`. The eval and eval bar still show; the best move
and engine lines stay hidden until the game ends.

## Recording test material

Every `play` session records all frames to `sessions/<date>_<time>/`. To score one later against
the moves that were really played, save those moves as a PGN and run:

```powershell
.\.venv\Scripts\python run.py replay sessions\<date>_<time> --pgn my_game.pgn
```

`record` saves plain video instead; `replay` accepts video files too (it then finds the board in
the first frame, so start the video on the starting position).
