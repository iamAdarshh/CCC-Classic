"""Level 6 optimizer v2: full-board hit counts, so same-line antiparallel beams (whole chords) are usable.

Usage: python opt2.py <input> <output> [config indices, comma separated]
"""
import os
import sys

import numpy as np

import opt
from opt import OPTS, SYM, Board, l5, l6

# combos = (first three emitters) x (last three emitters), 64 x 64, same order as OPTS
IDX3 = np.array([(a, b, c) for a in range(4) for b in range(4) for c in range(4)])


class Board2(Board):
    def __init__(self, r, pattern):
        super().__init__(r, pattern)
        nc = len(self.cells)
        # LINE[e, p]: membership vector of emitter e's beam at ring position p
        self.LINE = np.zeros((6, self.L, nc), dtype=np.int8)
        for e in range(6):
            c = e % 3
            for p in range(self.L):
                k = self.KEY[e, p]
                if k >= 0:
                    self.LINE[e, p] = self.CK[c, :nc] == k
        self.P = self.pmask[:nc]
        self.F = self.status[:nc] == 0


def hits(parts):
    """parts: (6,4,nc) per-emitter per-option membership -> (64,64,nc) hit counts."""
    A = parts[0][IDX3[:, 0]] + parts[1][IDX3[:, 1]] + parts[2][IDX3[:, 2]]
    B = parts[3][IDX3[:, 0]] + parts[4][IDX3[:, 1]] + parts[5][IDX3[:, 2]]
    return A[:, None, :] + B[None, :, :]


def evaluate2(b, pos, act, built, weights):
    L = b.L
    nc = len(b.cells)
    npos = np.stack([pos, (pos + 1) % L, (pos - 1) % L, pos], axis=1)
    nact = np.stack([act, act, act, 1 - act], axis=1)
    lines = b.LINE[np.arange(6)[:, None], npos]  # (6,4,nc)
    on = lines * nact[:, :, None].astype(np.int8)
    bt = built[:nc]
    unb = b.P & ~bt
    H = hits(on)
    two = H == 2
    if len(weights) > 5 and weights[5]:
        wf = weights[5]
        bad = ((H >= 3) & bt & b.P).any(-1)
        gain = (two & unb).sum(-1) - wf * (two & b.F & ~bt).sum(-1) + wf * ((H >= 3) & b.F & bt).sum(-1)
    elif len(weights) > 4 and weights[4]:
        bad = (two & b.F).any(-1)
        gain = (two & unb).sum(-1) - ((H >= 3) & bt).sum(-1) * weights[4]
    else:
        bad = (two & b.F).any(-1) | ((H >= 3) & bt).any(-1)
        gain = (two & unb).sum(-1)
    HL = hits(lines)
    twoL = HL == 2
    latent = (twoL & unb).sum(-1) - (twoL & b.F).sum(-1) * weights[3]
    wg, wl, wc = weights[:3]
    cov = 0
    if wc:
        cov = (lines.astype(np.int32) * unb).sum(-1)  # (6,4)
        cov = (cov[0][IDX3[:, 0]] + cov[1][IDX3[:, 1]] + cov[2][IDX3[:, 2]])[:, None] + \
              (cov[3][IDX3[:, 0]] + cov[4][IDX3[:, 1]] + cov[5][IDX3[:, 2]])[None, :]
    score = wg * gain + wl * latent + wc * cov + np.random.random(gain.shape) * 0.01
    score[bad] = -1e18
    return score.reshape(-1), gain.reshape(-1)


def apply2(b, pos, act, built, line):
    L = b.L
    for e, ins in enumerate(line):
        if ins == "F":
            pos[e] = (pos[e] + 1) % L
        elif ins == "B":
            pos[e] = (pos[e] - 1) % L
        elif ins == "T":
            act[e] ^= 1
    nc = len(b.cells)
    H = sum(b.LINE[e, pos[e]].astype(np.int32) for e in range(6) if act[e])
    if isinstance(H, int):
        H = np.zeros(nc, dtype=np.int32)
    bt = built[:nc]
    bt[(H == 2) & (b.P | b.F)] = True
    bt[(H >= 3) & (b.P | b.F)] = False
    if bt[b.P].all() and not bt[b.F].any():
        bt[b.P] = False
        return True
    return False


def plan2(r, pattern, count, weights, stall_limit, seed):
    np.random.seed(seed)
    b = Board2(r, pattern)
    g = b.g
    pos = np.array(g.start, dtype=np.int64)
    act = np.zeros(6, dtype=np.int64)
    built = np.zeros(len(b.cells) + 1, dtype=bool)
    prog, done, stall = [], 0, 0
    limit = 40 * b.pcount * count + 1000
    while done < count:
        if len(prog) > limit:
            return None
        if stall >= stall_limit:
            bset = {b.cells[i] for i in np.nonzero(built[:-1])[0]}
            res = l6.next_cell(g, list(map(int, pos)), list(map(int, act)), b_P(b), bset)
            if res is None:
                return None
            lines = res[0]
            stall = 0
        else:
            score, gain = evaluate2(b, pos, act, built, weights)
            k = int(np.argmax(score))
            lines = [[SYM[o] for o in OPTS[k]]]
            stall = 0 if gain[k] > 0 else stall + 1
        for line in lines:
            prog.append(line)
            if apply2(b, pos, act, built, line):
                done += 1
                stall = 0
                if done >= count:
                    break
    return prog


def b_P(b):
    return {b.cells[i] for i in np.nonzero(b.P)[0]}


CONFIGS = [
    ((1000, 30, 1, 0.5), 6),
    ((1000, 30, 1, 1.0), 6),
    ((1000, 60, 2, 1.0), 8),
    ((1000, 15, 0.5, 0.5), 6),
    ((1000, 30, 0, 1.0), 6),
    ((1000, 100, 1, 1.0), 10),
]


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    cfg = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else range(len(CONFIGS))
    items = opt.parse(open(inp).read())
    best = opt.read_progs(outp, len(items))
    for idx, (r, pat, count) in enumerate(items):
        if best[idx] is None or l5.run(r, pat, best[idx]) < count:
            best[idx] = l6.plan(r, pat, count)
        for ci in cfg:
            weights, stall = CONFIGS[ci]
            prog = plan2(r, pat, count, weights, stall, seed=ci * 1009 + idx)
            if prog is None or len(prog) >= len(best[idx]):
                continue
            if l5.run(r, pat, prog) >= count:
                best[idx] = prog
    for idx, (r, pat, count) in enumerate(items):
        assert l5.run(r, pat, best[idx]) >= count
    tmp = outp + ".tmp"
    with open(tmp, "w") as f:
        f.write("".join("\n".join(" ".join(l) for l in p) + "\n\n" for p in best))
    os.replace(tmp, outp)
    print(inp, "score", sum(len(p) for p in best))


if __name__ == "__main__":
    main()
