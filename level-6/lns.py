"""Restart search: keep a prefix of the best program, re-plan the rest with beam2, keep if shorter.

Usage: python lns.py <input> <shard_out.json> <i> <n> <width> <restarts> [look]
"""
import json
import sys

import beam2
import opt
import opt2
import polish

WEIGHTS = (1000, 30, 0, 0)


def improve(r, pat, count, prog, width, restarts, look, seed=0):
    b = opt2.Board2(r, pat)
    best = [list(l) for l in prog]
    S = polish.states(b, best)
    n = len(best)
    cuts = sorted({int(n * k / (restarts + 1)) for k in range(restarts + 1)})
    for t in cuts:
        if t >= len(best):
            continue
        S = polish.states(b, best)
        if S[t][3] >= count:
            continue
        room = len(best) - t - 1  # suffix must be strictly shorter to help
        suf = beam2.beam_plan2(r, pat, count, WEIGHTS, width, 8, 6, seed + t, look=look,
                               look_pool=4 * width, init=S[t], limit=room)
        if suf is None or t + len(suf) >= len(best):
            continue
        cand = best[:t] + suf
        if polish.count_subs(b, cand, count) >= count:
            best = cand
    return best


def main():
    inp, out, i, n, W, R = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7])
    look = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    outp = inp.replace("in_", "out_")
    items = opt.parse(open(inp).read())
    progs = opt.read_progs(outp, len(items))
    res = {}
    for idx, ((r, pat, count), prog) in enumerate(zip(items, progs)):
        if idx % n != i:
            continue
        new = improve(r, pat, count, prog, W, R, look, seed=idx)
        if len(new) < len(prog) and opt.l5.run(r, pat, new) >= count:
            res[idx] = new
        json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main()
