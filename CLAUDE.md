# CCC Classic contest workspace

Speed matters: rank = highest level reached, then total time. Be fast and pragmatic.

## Layout
- `levelN/` holds the downloaded files for level N (inputs `*.in`, an example with its expected output, the task description).
- Write solutions as `levelN/solve.py`. Outputs go next to inputs as `<same name>.out`.
- `run.py N` runs `levelN/solve.py` on every `*.in` in `levelN/`; `run.py N --example` checks only the example against its expected output.

## Workflow for each level
1. Read the task description fully, including edge cases and the exact output format.
2. Write `levelN/solve.py` exposing `solve(text: str) -> str`. Levels build on each other: copy the previous level's solve.py forward and extend it.
3. Run `python run.py N --example` until it says OK. Only then run `python run.py N`.
4. Report which outputs were produced, runtimes, and anything suspicious.

## Rules of thumb
- Output must match exactly. No debug prints in the returned string.
- Check complexity against the input sizes before running: later levels usually grow the inputs until brute force is too slow.
- Watch for recursion limits (set sys.setrecursionlimit or use iteration) and off-by-one errors in grid/coordinate tasks.
- Never overwrite a previous level's solve.py.
