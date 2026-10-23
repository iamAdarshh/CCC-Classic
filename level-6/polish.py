"""Polish level 6 programs: drop single lines or merge neighbouring lines, keeping a change only if it replays.

Usage: python polish.py <input> <output> [i n]   (optional shard: only patterns with idx % n == i, writes JSON)
"""
import json
import os
import sys

import numpy as np

import opt
import opt2


def count_subs(b, prog, need):
    """Fast replay with full-board hit counts; returns number of submissions (stops at need)."""
    nc = len(b.cells)
    pos = np.array(b.g.start, dtype=np.int64)
    act = np.zeros(6, dtype=np.int64)
    grid = np.zeros(nc, dtype=bool)  # matter anywhere on the board
    done = 0
    L = b.L
    P = b.P
    ring = b.F
    for line in prog:
        for e, ins in enumerate(line):
            if ins == "F":
                pos[e] = (pos[e] + 1) % L
            elif ins == "B":
                pos[e] = (pos[e] - 1) % L
            elif ins == "T":
                act[e] ^= 1
        H = np.zeros(nc, dtype=np.int32)
        for e in range(6):
            if act[e]:
                H += b.LINE[e, pos[e]]
        grid[H == 2] = True
        grid[H >= 3] = False
        if grid[P].all() and not grid[ring].any():
            grid[P] = False
            done += 1
            if done >= need:
                return done
    return done


def merge_lines(a, c):
    out = []
    for x, y in zip(a, c):
        if x != "-" and y != "-":
            return None
        out.append(x if x != "-" else y)
    return out


def polish(r, pat, count, prog):
    b = opt2.Board2(r, pat)
    prog = [list(l) for l in prog]
    improved = True
    while improved:
        improved = False
        i = len(prog) - 1
        while i >= 0:
            cand = prog[:i] + prog[i + 1:]
            if count_subs(b, cand, count) >= count:
                prog = cand
                improved = True
            elif i + 1 < len(prog):
                m = merge_lines(prog[i], prog[i + 1])
                if m is not None:
                    cand = prog[:i] + [m] + prog[i + 2:]
                    if count_subs(b, cand, count) >= count:
                        prog = cand
                        improved = True
            i -= 1
    return prog


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    shard = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else None
    items = opt.parse(open(inp).read())
    progs = opt.read_progs(outp, len(items))
    res = {}
    for idx, ((r, pat, count), prog) in enumerate(zip(items, progs)):
        if shard and idx % shard[1] != shard[0]:
            continue
        new = polish(r, pat, count, prog)
        if len(new) < len(prog) and opt.l5.run(r, pat, new) >= count:
            res[idx] = new
        if shard:
            json.dump(res, open(sys.argv[5], "w"))
    if not shard:
        for idx, p in res.items():
            progs[idx] = p
        tmp = outp + ".tmp"
        with open(tmp, "w") as f:
            f.write("".join("\n".join(" ".join(l) for l in p) + "\n\n" for p in progs))
        os.replace(tmp, outp)
        print(inp, "score", sum(len(p) for p in progs))


if __name__ == "__main__":
    main()


# ---- window replacement: replace w lines by w-1 lines reaching an equivalent state ----

import itertools as _it

SEQS = {k: list(_it.product("-FBT", repeat=k)) for k in (1, 2, 3)}


def states(b, prog):
    nc = len(b.cells)
    pos = np.array(b.g.start, dtype=np.int64)
    act = np.zeros(6, dtype=np.int64)
    grid = np.zeros(nc, dtype=bool)
    done = 0
    out = [(pos.copy(), act.copy(), grid.copy(), done)]
    for line in prog:
        pos, act, grid, sub = tick(b, pos, act, grid, line)
        done += sub
        out.append((pos.copy(), act.copy(), grid.copy(), done))
    return out


def tick(b, pos, act, grid, line):
    pos, act, grid = pos.copy(), act.copy(), grid.copy()
    for e, ins in enumerate(line):
        if ins == "F":
            pos[e] = (pos[e] + 1) % b.L
        elif ins == "B":
            pos[e] = (pos[e] - 1) % b.L
        elif ins == "T":
            act[e] ^= 1
    H = np.zeros(len(b.cells), dtype=np.int32)
    for e in range(6):
        if act[e]:
            H += b.LINE[e, pos[e]]
    grid[H == 2] = True
    grid[H >= 3] = False
    sub = 0
    if grid[b.P].all() and not grid[b.F].any():
        grid[b.P] = False
        sub = 1
    return pos, act, grid, sub


def emitter_options(b, p0, a0, p1, a1, k):
    opts = []
    for seq in SEQS[k]:
        d = sum(1 if s == "F" else -1 if s == "B" else 0 for s in seq)
        t = sum(s == "T" for s in seq) % 2
        if (p0 + d) % b.L == p1 and (a0 ^ t) == a1:
            opts.append(seq)
    return opts


def try_window(b, S, i, w, max_combos=20000):
    k = w - 1
    p0, a0, g0, d0 = S[i]
    p1, a1, g1, d1 = S[i + w]
    if d1 != d0:
        return None
    per = [emitter_options(b, int(p0[e]), int(a0[e]), int(p1[e]), int(a1[e]), k) for e in range(6)]
    n = 1
    for o in per:
        n *= len(o)
        if n == 0 or n > max_combos:
            return None
    need = g1[b.P]
    for combo in _it.product(*per):
        lines = [[combo[e][t] for e in range(6)] for t in range(k)]
        pos, act, grid = p0, a0, g0
        ok = True
        for line in lines:
            pos, act, grid, sub = tick(b, pos, act, grid, line)
            if sub or grid[b.F].any():
                ok = False
                break
        if ok and (grid[b.P] | ~need).all():
            return lines
    return None


def polish_windows(r, pat, count, prog, widths=(2, 3)):
    b = opt2.Board2(r, pat)
    prog = [list(l) for l in prog]
    changed = True
    while changed:
        changed = False
        S = states(b, prog)
        for w in widths:
            i = 0
            while i + w <= len(prog):
                rep = try_window(b, S, i, w)
                if rep is not None:
                    cand = prog[:i] + rep + prog[i + w:]
                    if count_subs(b, cand, count) >= count:
                        prog = cand
                        S = states(b, prog)
                        changed = True
                        continue
                i += 1
    return prog
