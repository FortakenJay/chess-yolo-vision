# Series log

A Freenove ESP32-WROVER with an OV2640 watches a roll-up vinyl set from about 75° overhead. That module brings up its own open Wi-Fi, `ChessCam`, and sits at `192.168.4.1`. Frames arrive around one per second, 800×600, dim and green. The PC records the game, asks Stockfish for an evaluation, and writes a PGN. Which ply was played comes from the pictures and the rules.

Dates on the commits are 2026-10-16 and 2026-10-17. Author on each one: jih bin luo liu. Remote: `https://github.com/FortakenJay/chess-yolo-vision.git`, branch `main`. Everything below is on GitHub, pushed 2026-10-21; the fix commit of each episode is tagged `ep-3` … `ep-10`.

| Commit | Tag | What the afternoon showed |
|---|---|---|
| `da52aaa` |  | `.gitattributes` only |
| `000feda` |  | First full tree |
| `954ccf9` |  | Piece model abandoned; difference-matching takes over |
| `61ed1f5` | `ep-3` | Camera drift followed instead of trusted from disk |
| `7bbbb24` | `ep-4` | Takebacks, texture channel, exposure offset |
| `1c646cf` | `ep-5` | Faint e7–e5 kept |
| `ef2d7ac` | `ep-6` | Forearm beside the vinyl |
| `a1a87ba` | `ep-7` | Sleeve over h1, dots, Stockfish lookup |
| `615119e` | `ep-8` | Shaded empties, a pawn stood back up |
| `ec98211` | `ep-9` | A stray Ke7, an R that resigned, an opening book |
| `85bafe7` |  | Replays that behave like the person at the board |
| `5d7e5cf` | `ep-10` | Standard tooling and a cleanup that changes nothing on the board |

Live play writes every JPEG under `sessions/<date>_<time>/`. `run.py replay` feeds those JPEGs through the same `session.py` live play uses. The number that decides a fix is **wrong accepts**: a continuation taken with no question that was not the one played. Zero is the target. Replay exits 2 otherwise. Each regression test is written so the code from the day before fails it.

---

## 0. Four cameras in one folder

`000feda` landed the whole workbench: `src/` running YOLO on a phone stream and smoothing FEN strings into Supabase, `code/` with an ESP32 sketch plus an ArUco generator, `ios_phone_streamer/`, two Expo trees (`mobile/`, `mobile_downgrade/`), and `web_streamer/`. Marker sheets were already rendered. There is no printer, so those sheets were never going on the table. `da52aaa`, the commit under it, is two lines of `.gitattributes`.

Nothing in that tree is a finished reader. It is the pile an episode can open on: phone, ESP32, printed-marker plan, piece detector. `954ccf9` deletes 173 paths and keeps one camera.

---

## 1. Nine to fifteen boxes

`954ccf9`. Commit message: "white working black contrast failure. base model working."

A starting-position JPEG went through `models/pieces.pt`. Nine to fifteen boxes came back. Thirty-two pieces were on the vinyl. Most boxes said knight.

Those weights are a 13-class YOLO detector, trained on Roboflow photographs at 800 px. Class 0 is a colourless `bishop` left over from the dataset. White plastic cleared the pale squares often enough to look like a product. Black plastic on the dark squares occupied the same brightness band as the vinyl, so the detector had little to outline. Lowering the confidence cutoff and enlarging the network input left the hit count inside that same 9–15 range.

Naming every piece on every frame stopped being the design. The replacement uses the rules. `python-chess` holds the position and enumerates legal continuations. `geometry.py` recovers a pinhole camera from the homography (the mapping that flattens the photo into an 800×800 top-down grid, 100 px a square) and draws each piece as a tapered cylinder. Heights, in squares: pawn 0.85, knight 1.2, bishop 1.4, rook 1.0, queen 1.55, king 1.75. From this angle a crown lands in different pixels than its base, and a piece on the near rank leans past the edge of the roll-up. `reader.py` subtracts the last accepted picture from the current settled one. `matcher.py` ranks legal continuations by how well each silhouette covers that difference.

Promotions stay with the detector. Queen, rook, bishop, and knight leave the same before-and-after silhouette when a pawn reaches the last rank; only the new shape separates them. Under-promotions are first scored as queens, then corrected.

Same commit drops the phone bridge, the iOS streamer, both Expo apps, the web streamer, the ArUco sheets, and Supabase. Games are PGN files in `exports/`. The evaluation bar is drawn in the Board window. `docs/original-spec.md` keeps the earlier plan and marks it historical.

QA row, 2026-10-17: "YOLO piece model on a starting-position frame."

---

## 2. Digits backwards, then 1.e4 treated as glare

Same rebuild, once it met the vinyl.

Printed coordinates on the ranks read in reverse. The OV2640 was delivering a vertical mirror. `camera.vflip` in `config.yaml` is sent over HTTP at startup. The program then reads `/status` and the size of a decoded JPEG, because this camera library numbers frame sizes differently from one Arduino package to the next. After the flip, the digits read normally. Orientation from piece colours, tried on all four rotations, came out right on every run that day: contrast 80–83 for the true rotation, about −85 or ~0 for the other three. A mirrored mapping is refused earlier, when corners are ordered clockwise, so a bad rotation can be wrong by 90° and cannot be a mirror.

1.e4 was the top-ranked continuation and still produced a question. About 68% of the difference was filed as unexplained. Auto-exposure had lifted the whole picture, and a crease in the roll-up had slid by a few pixels.

Two filters separate that from a relocation. An exposure gain is fitted so a global brightness shift drops out of the difference. A leftover counts when it forms blobs about the size of a piece. Speckle and hairlines are stripped with a morphological opening roughly a third of a square wide: a crease disappears, a second nudged piece stays.

Replaying those same JPEGs: tier 1, margin 3,236, 4% unexplained.

Tiers, once, because later episodes use the names. Tier 1 accepts on one settled picture when the score, the gap to second place, and the leftover blobs are all inside limits. Tier 2 averages a few more settled pictures (three re-reads; at this frame rate that is a couple of seconds) and accepts on a smaller gap. Tier 3, off unless asked, sends the before/after pair and the top three continuations to a vision model that must answer with one of those three. Tier 4 draws three numbered arrows and waits for a key. Hands are supposed to leave the vinyl before the picture settles, or the difference includes fingers.

Forty-seven settled frames, 45 seconds, nothing relocated: 0 false accepts. Every candidate sat near −2,700.

Other failures closed inside this same rebuild, each one visible in `docs/testing.md`, none with its own commit:

A reader thread died when two JPEGs arrived inside one Windows clock tick. Timestamps were being treated as unique. The grabber now keeps only the newest JPEG and survives a repeated tick.

Automatic corner finding gave up before this slow camera produced its first picture. It now waits for a real frame.

Lighting drift opened a "which move?" prompt. A small residual with no standout candidate is ignored. The later cutoff for that ignore is 6,000 px.

An uncertain continuation was discarded when the next relocation started during the tier-2 re-reads. The decision now uses whatever re-reads have already arrived.

Writing the PGN crashed when `exports/` was missing. The folder is created.

A cable draped across the vinyl was recorded as h3. Once blob size and the silhouette test were in place, that session records nothing.

Calibration itself, on this overhead view with the pieces already set up, fitted the inner corners to 0.79–0.85 px of error. A square is 100 px in the flattened grid. Occupancy on the starting position mismatched on 0 squares. Stockfish 18 answered +0.40, best move e4.

A synthetic 16-ply game at about 1 fps, with a fake hand between plies and exposure wander: 16/16, 0 wrong accepts, 94% automatic (14 tier 1, one tier 2, one question). Impatient players, same scene: 16/16, 0 wrong accepts.

---

## 3. The grid from yesterday, a square to the left

`61ed1f5`, on GitHub. New file: `chessreader/tracking.py`.

Calibration JSON was loaded and reused. Between runs someone bumped the camera. The stored grid then sat 25–80 px off the vinyl. On this sensor a square is about 55 px in the raw picture, so the overlay had walked by as much as one square. No continuation could be matched, because every silhouette was drawn on the wrong cells.

A synthetic game with the camera nudged three times mid-game accepted 3 of 16 plies.

Each new game now finds the vinyl again from the starting position. Ranks 3–6 are empty, so a 7×3 block of inner corners is visible even with pieces on the back ranks; that block pins the whole grid. Both the fresh detection and the file on disk are oriented by piece colour, and the tighter colour fit wins. Tall pieces hide corners. A detection that snaps a rank off would otherwise beat a good saved grid. On the real set the fresh fit wins, for example 87 against 50.

Once play has started, every settled picture is matched to the previous one with ORB features. The estimated shift is applied to the calibration and to the reference JPEG together, so the silhouettes and the difference stay in the same place. A shift too large to follow, or a picture where the set is covered, pauses the game. Press **C**. Pieces stay put. Log line when the small case works: `Camera moved ... followed it`.

Same synthetic game afterward: 16/16, 0 wrong accepts.

---

## 4. Put the pawn back, and the hand that biased the exposure

`7bbbb24`.

Returning a piece by hand never entered the rules. The matcher scored forward legal continuations, and the PGN walked away from the set. On the 09:55 session the recorded position disagreed with the settled picture on 196 of 290 frames. Sliding the h-pawn from h5 back to h7 once outscored every honest continuation and looked like Qxh5.

A takeback repaints the pixels the previous ply painted. That previous silhouette is scored alongside the forward list. If it leads every legal next ply by `t2_margin`, and both of its ends show up in the difference, the ply is undone. No key. The comparison runs even when some forward ply looks confident, which is what caught the fake queen capture. Log line: `was taken back`.

Exposure gain had its own bug. Gain was clipped to a legal range and the offset from that clipped fit was kept. A hand inside the reference JPEG then made the entire playing surface look different.

The offset from a clipped gain is discarded.

Question handling kept a dirty reference. The JPEG saved when the arrows appeared often still contained fingers. After the answer, that JPEG became the new reference, so the following comparison started dirty.

That JPEG is provisional. If the next settled picture is not a clean continuation from it, the reference becomes the vinyl as it looks once the hand is gone. While arrows are on screen, a later tier-1 reading is taken without a keypress, and a takeback closes the question. Tracking runs again after the answer, in case the camera shifted during the pause.

09:55, after the three fixes: 0 of 290 settled frames disagreed. Each returned ply is in the log.

---

## 5. A black pawn that barely dents the dark square

Two commits, both local. The first made the pawn visible. The second stopped throwing the pawn away.

### Texture, `7bbbb24`

Brightness subtraction misses a black pawn travelling from one dark square to another. On this vinyl the pawn and the square are nearly the same grey, so the difference image stays quiet even though a carved piece arrived. Wood grain and the moulded rim do change the local flicker of the grey.

A second channel measures that flicker: how wide the brightness runs inside a small neighbourhood. It is compared with the reference picture's own neighbourhood range. A square boundary nudged 1–2 px produces a thin edge in both pictures and drops out. A pawn's surface does not.

### The score floor, `1c646cf`

On the 10:16 session e7–e5 was the leading candidate on every settled picture and still died under `min_score`. The program filed it as lighting drift. 43 of 75 settled frames held the wrong position.

One expected rate, `q_in`, had been used for every pixel: the fraction of a silhouette that ought to differ if that ply happened. White-on-pale and black-on-dark do not differ by the same fraction. At the start of a game the reader measures contrast from the square colours and piece colours it just learned. On this roll-up, black-on-dark measures about 12 and white-on-light about 40. Those become per-pixel rates: about 0.28 where black plastic meets a dark square, about 0.55 where white plastic meets a light one. A pixel's rate depends on what it showed before and what the ply would put there.

A weak leader with a positive score and a gap of at least `t2_margin` goes to the tier-2 re-reads. `min_score` moved from 1,000 to 500. On this camera a picture where nothing relocated scores about −800, with gaps under 50, so 500 still refuses "nothing happened." Clear plies (e4, Nf3, Bc4, O-O) land between 4,000 and 19,000, gaps 1,800–8,000. A black pawn, dark square to dark square, lands between 500 and 1,000, gaps 1,500–1,800. A dark-squared bishop such as f8–e7 can sit near 0 with a gap of 1,000–1,400, which is why the re-read path matters more than the floor.

Startup orientation also compares the grid found this minute with the one in the calibration file and keeps the better piece-colour fit. Hidden corners can snap a detection a rank off; the colour score stops that detection from winning.

10:16 afterward: those 43 mismatches gone, e5 accepted at tier 1. 09:55 stays at 0. Log line at game start: `black pieces on dark squares: contrast N, q_in Q`.

---

## 6. A forearm on the table, and a king that never left g1

`ef2d7ac`.

e5 and c5 led the ranking and still became questions. An arm resting on the table past the vinyl edge was counted as leftover difference, and leftover difference blocks tier 1. The same strip of pixels let a hand near the h1 corner pass as Kh1 at tier 1.

From this mount the tops of the taller pieces are what reach beyond the edge. Forearms occupy that strip the rest of the time.

Off-vinyl pixels may raise a ply's score by at most the amount its on-vinyl pixels already earned. A hand beside h1 cannot assemble Kh1 when h1 itself is quiet. Leftover difference is measured on the vinyl only. Blobs past the edge that the leader does not explain are reported separately as `off_board`. Above `max_off_board` (3,000 px) a hand is assumed and nothing is taken automatically. Smaller blobs out there, a sleeve lying flat, no longer count as unexplained, so they stop blocking a clean pawn push.

Sessions 09:55, 10:16, and 10:28: 0 mismatched settled frames. On the 10-ply game (e4 e5 Bc4 Nf6 Nf3 d6 Nc3 Be7 O-O c5) that is 0 of 527. c5 is tier 1. e5 and Be7 are tier 2. Kh1 does not appear. The regression scene uses an overhead camera like the real mount, with pieces at the rim leaning outward, and fails on the code from the day before.

---

## 7. Cloth on one corner

`a1a87ba`.

A sleeve lying over h1 was recorded as Rg1, and three times as Kh1, while the white king remained on g1. g1, the square the king would have left, showed 0.16–0.61 of the rate expected if a white piece had departed a dark square. Genuine plies that day, wherever the piece contrasts with its square, reached at least 0.97 at both ends. The session is 10:49: 20 plies, one takeback, 874 settled frames.

Both ends have to show up. For each continuation the matcher keeps the observed rate against `q_in` on the departure square and on the arrival square (both rook squares as well, when castling). Where the expected rate is at least `clear_end_q` (0.45), each of those squares must reach `min_end_ratio` (0.75) of what was expected. Faint ends stay outside that test. A real d6–d5 can move only about 0.27 of its expected rate on a dark square; applying 0.75 there would reject the pawn. Takebacks use the same pair of ends.

Two smaller faults rode along in this commit.

Dots in the Board window refreshed only at a sync check every 10 plies, so the diagram lagged the vinyl. They now refresh after every accepted ply and at the start of the game.

`engine.path` pointed at a Windows executable. On a Mac that file is absent, Stockfish never started, and the evaluation bar sat still. A missing path is replaced by a binary found on `PATH`, then in the usual Homebrew and Linux locations, then in `engines/` (git-ignored). Log line when it works: `Stockfish ready`. When it does not: `Stockfish not found ...; analysis off`.

09:55, 10:16, 10:28, 10:49 afterward: 0 mismatched settled frames. 10:49 ends on the plies that were played. Rg1 and Kh1 are gone. Questions in that session coincide with a hand at the vinyl.

---

## 8. Shade on a5, and a pawn stood upright

`615119e`.

Dots for empty versus occupied were mostly a colour test. Shaded empty cells at the rim, a5 and a6, passed as occupied. Measuring the real vinyl: empty cells have a brightness spread of about 0.2–2.8; occupied cells, black on dark included, sit at 5.6 and above.

Spread is now the occupancy test, taken over 60% of the cell so a shadow in one corner cannot fill the statistic. Colour runs after spread says the cell is occupied, and its job is white versus black. A dot under 60% confidence is omitted. Both cues are learned from the starting position in whatever light the session has; **N** starts a game and learns them again. A sleeve or a hard shadow on a cell reads as occupied because the spread is physically there.

Separately, a b2 pawn that began the session knocked over was stood back up. One cell differed. The prompt offered `1) b4  2) b3  3) h3`. Session 11:10: nothing had been played.

An in-place adjustment has its own silhouette, the piece outline widened a little. If that silhouette beats every legal continuation by half of `t2_margin`, and the likeliest relocation of that piece would have been obvious on the destination cell but the destination stayed quiet, the picture is accepted as the new reference and no arrows appear. Log line: `adjusted in place`. A faint destination is ineligible for the "stayed quiet" clause. Black arriving on a dark square is allowed to look quiet, so a genuine faint ply cannot be filed as someone tidying a piece.

When no legal continuation fits and at least `wait_off_board` (1,000 px) of the residual lies past the edge, the program waits. On-screen text: "Something is resting at the board's edge; waiting." The earlier behaviour was a question.

09:55, 10:16, 10:28, 10:49, 11:10: 0 mismatched settled frames. 11:10 asks nothing, 0 of 83, and a5/a6 stay blank.

---

## 9. Ke7 by accident, an R that resigned, and an opening book

`ec98211`.

Session 11:28 (`sessions/20261017_112825`). 1.e4 e5 2.Nf3, then a stray 2...Ke7, taken at tier 1 as it should be: a legal king step is a legal king step. 3.d4 exd4 4.Qd2 followed; Qd2 was returned by hand and undone. The next settled picture asked `1) Nfd2  2) Ng1  3) Nxd4`: three crossing arrows drawn over the last-move arrow. **R** was pressed to start over. R meant resign. Log: `Game over (black)`.

R now resets. The game so far is written as an unfinished PGN (`*`), and a new one starts on the spot if the starting position is back on the vinyl, otherwise on **S**. "Is it back?" cannot be `start_game`'s colour check alone, which would start happily with a stray e4 still standing; the square readings are compared with the starting position and with the current one, and the start has to fit better. Resign moved to **G**. A reset inside the same second used to give the new PGN the old file's name; the second file now gets `_2`.

The arrows: the last ply is a translucent tint on its two squares. While a question is open the tint goes, and each candidate becomes a thin translucent arrow with a numbered badge tucked into a corner of the destination, away from the piece. The first candidate is drawn a little stronger.

Opening theory: `chessreader/openings.py` and `chessreader/data/openings.tsv`. 1,370 named lines, ECO A00–E99, Lichess-style names, 2,218 positions. Written for the repo rather than downloaded; the tests replay every line for legality. Positions are keyed by FEN without move counters, so move order does not matter: the 10:28 game, 1.e4 e5 2.Bc4 Nf6 3.Nf3, is Petrov's Defense: Italian Variation (C42). The name is the deepest one reached and stays after the game leaves the book: 10:49 reads `C40 McConnell Defense   (out of book at move 3)`. Theory moves are ranked by how many book lines run through them; the Board window shows three, hidden in training mode like the best move. The PGN carries `ECO` and `Opening`. The book loads in about a quarter of a second at startup; a lookup takes about a millisecond.

The reset and arrows, and the book, were written by two subagents at the same time in separate files, then wired together: the `openings` settings, the line under the move list, and PGN tags that follow those settings.

09:55, 10:16, 10:28, 10:49, 11:10 afterward: 0 mismatched settled frames. Forty-eight tests.

`85bafe7` came out of writing the filming guide below. Without a PGN, `run.py replay` answered every question with "no move", the same as pressing **X**. A hand still in that picture became part of the reference, and everything after it went wrong: 09:55 stopped after d5, although the live game had followed every takeback. The open question now stays open, as it does for someone who just moves their hand away, and the table prints `taken back` rows and the final position. 09:55 then ends on e4 after four takebacks, 10:49 on all 20 plies, 11:28 on e4 e5 Nf3 Ke7 d4 exd4.

---

## 10. Nothing moves on the vinyl

`5d7e5cf`, tag `ep-10`. No new reading behaviour; the episode is about whether the code can be trusted to stay that way.

The tools first, since no Python equivalent of a React doctor exists as a skill: **ruff** (lint and format), **mypy** (types), **vulture** (dead code), **radon** (complexity), wired into `pyproject.toml` and a `pre-commit` config that runs the venv's own copies. The first survey: 772 ruff findings (593 of them long lines), 25 of 30 files off-format, 69 type errors, a handful of dead functions, and four functions past any sane complexity: `run_play` at 47 branches, `run_replay`, `render_board`, `GameSession.process`.

Before touching anything, a baseline: every one of the 12 recorded sessions replayed and saved, and the frame-by-frame checker that had judged every fix so far moved into the repo as `tools/eval_sessions.py`, with the truth for five games in `tools/session_timelines.json`.

Then the changes. `run_play` became `LiveGame`, one small method per key. `GameSession` grew a `Pipeline` (camera model, matcher, change reader, motion gate) built and rebuilt as one piece, and its per-frame work split into a handler per phase. Recording moved into `SessionRecorder`, which appends a row and closes the file each time. Most of the 69 type errors were one root cause: everything built at game start was `None` before it.

One slip worth showing. A script written to delete the dead functions computed the indentation of a method as zero and deleted from each one to the end of its class: `MotionGate.update`, `BoardCamera.render_labels`, `GameRecord.add_move`, the book's `lookup`. 22 tests failed at once. The four files were rebuilt from git plus the intended edits, and an AST comparison of every module against the last commit confirmed that only the dead names were gone.

Then Claude Code's `/simplify`: four review agents at once, one each for reuse, simplification, efficiency and altitude. They found the opening book cached in four stacked layers, the session file format read in three places, a no-op `close()` still called five times, an engine listener added on every reset and never removed (each old game kept alive in memory), `None` fallbacks that disagreed with each other, and an evaluator that decoded 3.3 GB of frames before processing the first. All fixed, except grouping the in-game state into one `Game` object, which is the next structural step.

The test of all of it: after every stage, the 12 sessions replayed **byte-identically** to the baseline and `eval_sessions.py` still reported 0 wrong frames in all five games. ruff, the formatter, mypy and vulture end clean; 51 tests. A test for the live key loop came out of a near-miss during review: binding **X** directly to the first session's method would have left it pointing at a game that **R** had replaced.

---

## How a fix gets believed

Take the `sessions/` folder from the hour it broke. Replay it. Count settled frames whose recorded position disagrees with the vinyl, and count accepts that disagree with the ply that was played. A PGN cannot describe a takeback, so those sessions are scored by the final position plus the `Move N` and `was taken back` lines, frame by frame, on every picture where the vinyl had gone quiet.

New drawings live in `tests/scene.py`: a low side view, and a near-overhead view with a textured table, darker squares, fainter rims. Noise and exposure wander are added. `tests/test_reader.py`, `tests/test_matcher.py`, `tests/test_replay.py`, `tests/test_engine.py`, `tests/test_openings.py`, `tests/test_display.py`, and `tests/test_play.py` hold the cases above. Threshold edits are replayed against the recorded sessions before they are trusted. Fifty-one tests, no camera required. `tools/eval_sessions.py` scores the recorded games frame by frame.

One row in `docs/testing.md` is still open. The 16-ply test game (`tests/data/test_game.pgn`, through 8.Nc3 O-O) has a synthetic replay at 16/16. The same game recorded on this vinyl and replayed against that PGN is marked pending.

A video description can link an episode as `https://github.com/FortakenJay/chess-yolo-vision/tree/ep-N`, or a single fix as `.../commit/<hash>`.

---

## Filming a stage

Every episode is one commit, and the session that exposed the fault is on disk. Check out the commit before the fix, replay the session, then check out the fix and replay again. Same JPEGs, two outcomes.

| Episode | Tag | Before | Fix | Session(s) | What changes on screen |
|---|---|---|---|---|---|
| 3 | `ep-3` | `954ccf9` | `61ed1f5` | `20261017_093628`, `20261017_093908` | Stale grid reads nothing (or a phantom h3 under the cable); after, the vinyl is found again |
| 4 | `ep-4` | `61ed1f5` | `7bbbb24` | `20261017_095558` | d5, d6, Nc6, h5 put back by hand: before, the PGN walks away; after, each `was taken back` |
| 5 | `ep-5` | `7bbbb24` | `1c646cf` | `20261017_101625` | e7–e5 dropped as lighting, then accepted at tier 1 |
| 6 | `ep-6` | `1c646cf` | `ef2d7ac` | `20261017_102849` | c5 asked and a phantom Kh1; after, c5 tier 1, no Kh1 |
| 7 | `ep-7` | `ef2d7ac` | `a1a87ba` | `20261017_104913` | Sleeve over h1 read as Rg1 and Kh1; after, a question while the sleeve is there |
| 8 | `ep-8` | `a1a87ba` | `615119e` | `20261017_111040` | Pawn stood up asks `b4 b3 h3`, dots on a5/a6; after, nothing asked, no stray dots |
| 9 | `ep-9` | `615119e` | `ec98211` | `20261017_112825` | Arrows and the R key are live-only; film them with `run.py play` |
| 10 | `ep-10` | `3c0aa6e` | `5d7e5cf` | any | Nothing may change: replay a session at both and `diff` the output; run `ruff check .` and `mypy` at both |

Each fix commit carries a tag, `ep-3` … `ep-10`, so an episode can be checked out by name; `ep-4^` is the code just before it.

```bash
git switch --detach ep-4^             # the code just before episode 4
.venv/bin/python run.py replay sessions/20261017_095558
git switch --detach ep-4              # the fix
.venv/bin/python run.py replay sessions/20261017_095558
git switch main
```

On GitHub the same tags are under **Tags**, and `https://github.com/FortakenJay/chess-yolo-vision/tree/ep-4` shows the repo as it was for episode 4.

`replay` prints each ply it took, its tier and margin, `taken back` rows, and the final position. Commits before `85bafe7` print only the plies taken, and their replay answers an open question with "no move", which can derail a session that the live code handled better. For an honest "before", show that day's live log too: `logs/run_<date>_<time>.log` is what the code at the table actually did. The log has every settled reading (`read: change ... best ... margin ...`), which is the frame-by-frame evidence. Without `--pgn`, a replay reports what it read; sessions with takebacks cannot be scored against a PGN, so compare the `Move N` and `was taken back` lines instead.

`sessions/`, `logs/`, `exports/` and `calib/` are git-ignored. The sessions above total about 480 MB of JPEGs. GitHub keeps the code for every episode, not the footage it was tested on. Copy `sessions/` somewhere safe (or attach a zip per episode to a GitHub release) before filming on another machine.
