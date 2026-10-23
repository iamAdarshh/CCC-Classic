# Level 6: Nanofabricator program optimizer

Level 6 is an optimization level. Each input gives patterns plus a required submission count. For each one we output an emitter program that submits the pattern at least that many times. The score is the total number of instruction lines, and lower is better. Only `2-medium`, `3-large` and `4-enormous` are scored.

## Score history (total lines across the three scored files)

| Version | medium | large | enormous | total |
|---|---|---|---|---|
| `solve.py`: one cell per tick, pair BFS | 26,269 | 22,321 | 20,408 | 68,998 |
| `opt.py`: greedy, all 6 emitters | 11,670 | 11,844 | 10,843 | 34,357 |
| `beam.py` W=64, partial (~1/3 of patterns done) | 10,118 | 9,989 | 9,229 | 29,336 |
| `beam.py` W=64, all patterns | 6,750 | 6,331 | 6,231 | 19,312 |
| + `beam2.py` W=64 and lookahead W=32 (merged) | 6,351 | 5,912 | 5,843 | 18,106 |
| friend's solver alone | 6,057 | 5,415 | 5,180 | 16,652 |
| merged with friend's files, polished, enormous lookahead W=64 | 6,035 | 5,351 | 5,102 | 16,488 |

Merging keeps the shortest verified program per pattern, so combining runs from any source can only lower the total.

## What did not help (tested on the first 15 patterns of `3-large`)

| Idea | Result |
|---|---|
| Whole-line macros: 2 opposite emitters on one line build the full line; "cutter" beams cancel pattern-adjacent cells with a 3rd hit (`macro.py`) | Never chosen over normal moves: switching off and moving into place costs more than the row gains |
| Allow destroying built cells (3 hits), scored as a penalty | 291–292 vs 292, no change |
| Temporary mistakes: allow matter next to the pattern if a later 3-hit clears it (`opt2.py` weight index 5) | Low penalty: search can't clear it, most runs invalid. High penalty: 306 vs merged best 276 |
| Polishing: delete or merge single lines, replace 2–3-line windows with 1 line fewer (`polish.py`) | 0 on our programs, 7 lines on the merged files |
| Restart: keep a prefix of the best program, beam-search the suffix (`lns.py`) | 0 improvement on the merged best |
| Heuristic weights (latent, coverage, forbidden penalty) | Almost no effect; width and lookahead are what matter |

## Rules the solvers rely on (from the PDFs for levels 4–6)

- The board is a hexagon. The six emitters move along the border ring: F moves clockwise, B counterclockwise, T toggles on/off, `-` does nothing. Each emitter starts in its own corner, switched off.
- Each tick: run the instruction line, then count beam hits per cell. A cell hit by exactly 2 beams becomes `M`; a cell hit by 3 or more becomes `O`. Then check for the pattern.
- The pattern is submitted (and removed) when every pattern cell is `M` and no matter is next to it. Matter elsewhere on the board is allowed. **The solvers use this:** stray matter is fine as long as it never lands next to the pattern.
- A program is valid when the number of submissions is at least the required count (rule from `sandbox-level-6.html`).

## Files

| File | Purpose |
|---|---|
| `solve.py` | Baseline planner, and shared geometry (`Geometry`, `next_cell` pair BFS). Imports the simulator from `../level-5/solve.py`. |
| `opt.py` | Greedy planner: scores all 4,096 instruction lines per tick with numpy (crossings of beam pairs). Also has the shared `Board`, `evaluate`, `parse` and `read_progs`. |
| `opt2.py` | Experiment: full-board hit counts so opposite emitters on the same line can build a whole row. It scored **worse** and isn't used. |
| `beam.py` | Beam search over ticks using `opt.evaluate`; falls back to the pair BFS if it makes no progress. |
| `beam2.py` | **Best solver.** Same beam search, but scored with full-board hit counts (`opt2.evaluate2`), which is ~5% better and faster. Options: `look` (one-step lookahead rescoring) and `init` (start from a given state). |
| `macro.py`, `lns.py`, `polish.py` | Experiments listed under "What did not help". `polish.py` also has the fast replay (`count_subs`). |
| `shard.py` | Runs `beam.py` on every n-th pattern of one input and writes `{index: program}` JSON. |
| `merge.py` | Merges shard JSON into an `out_*.txt` file. It keeps a pattern's new program only if it's shorter **and** passes the level 5 simulator. |
| `run_all.sh` | Shards all three scored inputs across the CPU cores, waits, then merges. |

Dependencies: Python 3 with numpy. The code also needs `../level-5/solve.py` (the simulator used for verification), so copy the whole `classic/` folder, or at least `level-5/solve.py` and `level-6/`.

## How the beam search works

- **State:** emitter positions, on/off flags, which pattern cells are built, and the number of submissions so far.
- **Each tick:** for every state in the beam, score all 4,096 instruction lines. Score = 1000 × new pattern cells + 30 × "latent" crossings (pattern cells where two current beam lines would meet if both were on) + coverage (unbuilt pattern cells on each emitter's line). Lines that would put matter next to the pattern, or hit a built cell with 3 or more beams, are discarded.
- **Next beam:** take the top `topk` lines per state, merge identical states, and keep the best `width` by cells built so far plus score.
- **Stall:** if no state makes progress for `stall` ticks, run the pair BFS from the best state to guarantee progress.
- **Finish:** stop at the first state that reaches the required count.

Benchmark on the first 15 patterns of `3-large` (lower is better):

| Run | Lines |
|---|---|
| greedy `opt.py`, best of 5 settings | 576 |
| beam W=16, K=8 | 345 (7 s) |
| beam W=32, K=8 | 335 (13 s) |
| beam W=64, K=8 | 308 (25 s) |

With `beam2.py` (full-board scoring): W=64 → 292, W=128 → 288 (width stops paying off), W=16 with lookahead 1.0 → 293 (as good as W=64 without it).

**Lookahead:** before cutting the beam to `width`, the best 4×width candidates are rescored with the most new cells their next tick could build (one extra evaluation each). This ranks states that are well set up above states that just built cells but are badly placed.

In `shard.py`, config `100` = `beam2` plain and `102` = `beam2` with lookahead, e.g. `./run_all.sh 32 8 <shards> 102`.

## Running on another machine

A GPU doesn't help: this is numpy on the CPU, one core per shard. Use a machine with many cores.

```bash
cd classic/level-6
python3 -m venv .venv && . .venv/bin/activate && pip install numpy
PY=python ./run_all.sh 64 8          # width 64, topk 8, cores/3 shards per file
PY=python ./run_all.sh 128 8 16      # wider beam, 16 shards per file
```

- To check progress mid-run (shards save after every pattern), merge whatever is there:
  `python merge.py in_level-6_3-large.txt out_level-6_3-large.txt shards/3-large.*.json`
- Runs can be repeated with other widths or configs. Merging only ever makes the files shorter, so results from several machines or runs can be combined. Copy the `shards/*.json` files into one folder and merge them all.
- `beam.CONFIGS` holds 4 weight/stall settings (index 0–3). Running other configs gives different programs, and merging keeps the best per pattern.

## Upload

Upload `out_level-6_2-medium.txt`, `out_level-6_3-large.txt` and `out_level-6_4-enormous.txt` on the level 6 page. Every program in them has been checked with the level 5 simulator.
