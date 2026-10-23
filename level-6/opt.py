"""Level 6 optimizer: greedy multi-emitter planner, keeps the best program per pattern.

Usage: python opt.py <input> <output> [seeds]
"""
import importlib.util
import itertools
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


l5 = load("l5", os.path.join(HERE, "..", "level-5", "solve.py"))
sys.modules["solve"] = l5  # level-6 solve.py imports from "solve"
l6 = load("l6", os.path.join(HERE, "solve.py"))

OPTS = np.array(list(itertools.product(range(4), repeat=6)), dtype=np.int64)  # 0 '-', 1 F, 2 B, 3 T
SYM = "-FBT"
PAIRS = [(i, j) for i in range(6) for j in range(i + 1, 6) if i % 3 != j % 3]
SAME = [(0, 3), (1, 4), (2, 5)]


class Board:
    def __init__(self, r, pattern):
        g = l6.Geometry(r)
        self.g = g
        self.L = g.L
        cells = [(y, x) for y, row in enumerate(g.grid) for x, ch in enumerate(row) if ch == "O"]
        self.cells = cells
        cid = {c: i for i, c in enumerate(cells)}
        nc = len(cells)
        raw = lambda c, y, x: (y, x - y, x + y)[c]
        kidx = []
        for c in range(3):
            vals = sorted({raw(c, y, x) for y, x in cells})
            kidx.append({v: i for i, v in enumerate(vals)})
        self.K = [len(k) for k in kidx]
        # KEY[e, p]: key index of emitter e's beam at ring position p, -1 if it misses the board
        self.KEY = np.full((6, self.L), -1, dtype=np.int64)
        for e in range(6):
            for p in range(self.L):
                k = g.key[e][p]
                if k is not None:
                    self.KEY[e, p] = kidx[e % 3][k]
        # CK[c, cell]: key index of the cell for class c; sentinel cell (-1) gets -2
        self.CK = np.full((3, nc + 1), -2, dtype=np.int64)
        for i, (y, x) in enumerate(cells):
            for c in range(3):
                self.CK[c, i] = kidx[c][raw(c, y, x)]
        # CROSS[(c1, c2)][k1, k2] -> cell id, -1 if none; last row/col is the sentinel for key -1
        self.CROSS = {}
        for c1 in range(3):
            for c2 in range(3):
                if c1 == c2:
                    continue
                t = np.full((self.K[c1] + 1, self.K[c2] + 1), -1, dtype=np.int64)
                for i in range(nc):
                    t[self.CK[c1, i], self.CK[c2, i]] = i
                self.CROSS[c1, c2] = t
        P = {(y, x) for y, row in enumerate(pattern) for x, ch in enumerate(row) if ch == "M"}
        self.P = P
        nbr = {(y + dy, x + dx) for y, x in P for dy, dx in l5.DIRS} - P
        # status: 0 forbidden (touches pattern), 1 junk allowed, 2 pattern
        self.status = np.ones(nc + 1, dtype=np.int64)
        for c, i in cid.items():
            if c in P:
                self.status[i] = 2
            elif c in nbr:
                self.status[i] = 0
        self.pmask = self.status == 2
        self.pcount = int(self.pmask.sum())
        # line membership of pattern cells for coverage heuristic
        self.pidx = np.nonzero(self.pmask)[0]


def evaluate(b, pos, act, built, weights):
    L = b.L
    npos = np.stack([pos, (pos + 1) % L, (pos - 1) % L, pos], axis=1)  # (6,4)
    nact = np.stack([act, act, act, 1 - act], axis=1)
    lines = b.KEY[np.arange(6)[:, None], npos]  # (6,4)
    keys = np.where(nact == 1, lines, -1)
    KC = keys[np.arange(6), OPTS]  # (4096,6)
    LC = lines[np.arange(6), OPTS]
    pc = np.empty((len(OPTS), len(PAIRS)), dtype=np.int64)
    pl = np.empty_like(pc)
    for k, (i, j) in enumerate(PAIRS):
        t = b.CROSS[i % 3, j % 3]
        pc[:, k] = t[KC[:, i], KC[:, j]]
        pl[:, k] = t[LC[:, i], LC[:, j]]
    H = np.zeros(pc.shape, dtype=np.int64)
    for e in range(6):
        H += KC[:, e][:, None] == b.CK[e % 3][pc]
    st = b.status[pc]
    bt = built[pc]
    bad = ((H == 2) & (st == 0)) | ((H >= 3) & bt)
    valid = ~bad.any(axis=1)
    for a, c in SAME:
        valid &= ~((KC[:, a] >= 0) & (KC[:, a] == KC[:, c]))
    newp = (H == 2) & (st == 2) & ~bt
    gain = newp.sum(axis=1)
    # latent: crossings of current lines (ignoring on/off) on unbuilt pattern cells
    latent = ((b.status[pl] == 2) & ~built[pl]).sum(axis=1)
    # coverage: unbuilt pattern cells on each emitter's line
    unb = b.pmask & ~built
    cov = np.zeros(len(OPTS))
    for e in range(6):
        c = e % 3
        cnt = np.bincount(b.CK[c][unb], minlength=b.K[c] + 1)
        cnt = np.append(cnt[: b.K[c]], 0)  # key -1 -> 0
        cov += cnt[LC[:, e]]
    wg, wl, wc = weights
    score = wg * gain + wl * latent + wc * cov + np.random.random(len(OPTS)) * 0.01
    score[~valid] = -1e18
    return score, gain, pc, H


def apply_line(b, pos, act, built, line):
    """Execute one tick on our own state; returns new built and whether pattern was submitted."""
    L = b.L
    for e, ins in enumerate(line):
        if ins == "F":
            pos[e] = (pos[e] + 1) % L
        elif ins == "B":
            pos[e] = (pos[e] - 1) % L
        elif ins == "T":
            act[e] ^= 1
    keys = [int(b.KEY[e, pos[e]]) if act[e] else -1 for e in range(6)]
    hits = {}
    for i, j in PAIRS:
        if keys[i] < 0 or keys[j] < 0:
            continue
        c = int(b.CROSS[i % 3, j % 3][keys[i], keys[j]])
        if c < 0 or c in hits:
            continue
        hits[c] = sum(1 for e in range(6) if keys[e] >= 0 and keys[e] == b.CK[e % 3, c])
    for c, h in hits.items():
        if b.status[c] == 2:
            if h == 2:
                built[c] = True
            elif h >= 3:
                built[c] = False
    if built[: len(b.cells)][b.pmask[: len(b.cells)]].all():
        built[:] = False
        return True
    return False


def greedy_plan(r, pattern, count, weights, stall_limit, seed):
    np.random.seed(seed)
    b = Board(r, pattern)
    g = b.g
    pos = np.array(g.start, dtype=np.int64)
    act = np.zeros(6, dtype=np.int64)
    built = np.zeros(len(b.cells) + 1, dtype=bool)
    prog = []
    done = 0
    stall = 0
    limit = 40 * b.pcount * count + 1000
    while done < count:
        if len(prog) > limit:
            return None
        if stall >= stall_limit:
            Pset = b.P
            bset = {b.cells[i] for i in np.nonzero(built[:-1])[0]}
            res = l6.next_cell(g, list(map(int, pos)), list(map(int, act)), Pset, bset)
            if res is None:
                return None
            lines, _ = res
            stall = 0
        else:
            score, gain, _, _ = evaluate(b, pos, act, built, weights)
            k = int(np.argmax(score))
            lines = [[SYM[o] for o in OPTS[k]]]
            stall = 0 if gain[k] > 0 else stall + 1
        for line in lines:
            prog.append(line)
            if apply_line(b, pos, act, built, line):
                done += 1
                stall = 0
                if done >= count:
                    break
    return prog


def parse(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    n = int(lines[0])
    i = 1
    items = []
    for _ in range(n):
        r, rows, _, count = map(int, lines[i].split())
        items.append((r, lines[i + 1:i + 1 + rows], count))
        i += 1 + rows
    return items


def read_progs(path, n):
    if not os.path.exists(path):
        return [None] * n
    blocks = [b for b in open(path).read().split("\n\n") if b.strip()]
    progs = [[l.split() for l in blk.strip().split("\n")] for blk in blocks]
    return progs if len(progs) == n else [None] * n


CONFIGS = [((1000, 30, 1), 4), ((1000, 30, 1), 8), ((1000, 60, 2), 6), ((1000, 15, 0.5), 6)]


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    items = parse(open(inp).read())
    best = read_progs(outp, len(items))
    for idx, (r, pat, count) in enumerate(items):
        if best[idx] is None or l5.run(r, pat, best[idx]) < count:
            best[idx] = l6.plan(r, pat, count)
        for s in range(seeds):
            for weights, stall in CONFIGS:
                prog = greedy_plan(r, pat, count, weights, stall, seed=s * 101 + idx)
                if prog is None or len(prog) >= len(best[idx]):
                    continue
                if l5.run(r, pat, prog) >= count:
                    best[idx] = prog
    total = sum(len(p) for p in best)
    for idx, (r, pat, count) in enumerate(items):
        assert l5.run(r, pat, best[idx]) >= count
    with open(outp, "w") as f:
        f.write("".join("\n".join(" ".join(l) for l in p) + "\n\n" for p in best))
    print(inp, "score", total)


if __name__ == "__main__":
    main()
