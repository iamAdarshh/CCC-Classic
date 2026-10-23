"""Chord macros: build a whole line of the pattern in one tick.

Two antiparallel emitters on the same line give every cell of that line 2 hits. Pattern-adjacent
(forbidden) cells on the line are cancelled by "cutter" beams crossing there (3 hits -> no matter).
A macro switches everything off, moves the participants into place, then fires them together.
"""
import itertools
from collections import defaultdict

import numpy as np

import beam
import beam2
import opt
import opt2
from opt import OPTS, SYM, l6


def positions_for(b):
    pf = [defaultdict(list) for _ in range(6)]
    for e in range(6):
        for p in range(b.L):
            k = int(b.KEY[e, p])
            if k >= 0:
                pf[e][k].append(p)
    return pf


def ring_path(L, p, t):
    f = (t - p) % L
    bk = (p - t) % L
    return ["F"] * f if f <= bk else ["B"] * bk


def chord_macros(b, pf, node, min_cells, max_macros):
    nc = len(b.cells)
    bt = node.built[:nc]
    unb = b.P & ~bt
    out = []
    for c in range(3):
        a1, a2 = c, c + 3
        counts = np.bincount(b.CK[c, :nc][unb], minlength=b.K[c])
        for k in np.nonzero(counts >= min_cells)[0]:
            k = int(k)
            if not pf[a1][k] or not pf[a2][k]:
                continue
            online = b.CK[c, :nc] == k
            forb = np.nonzero(online & b.F)[0]
            others = [e for e in range(6) if e % 3 != c]
            if len(forb) > len(others):
                continue
            base = b.LINE[a1, pf[a1][k][0]].astype(np.int32) + b.LINE[a2, pf[a2][k][0]]
            best = None
            for perm in itertools.permutations(others, len(forb)):
                tgt = {a1: pf[a1][k][0], a2: pf[a2][k][0]}
                H = base.copy()
                ok = True
                for f, e in zip(forb, perm):
                    ps = pf[e].get(int(b.CK[e % 3, f]))
                    if not ps:
                        ok = False
                        break
                    tgt[e] = ps[0]
                    H += b.LINE[e, ps[0]]
                if not ok:
                    continue
                if ((H == 2) & b.F).any() or ((H >= 3) & bt).any():
                    continue
                gain = int(((H == 2) & unb).sum())
                cost = max(len(ring_path(b.L, int(node.pos[e]), t)) for e, t in tgt.items())
                if best is None or (gain, -cost) > (best[0], -best[1]):
                    best = (gain, cost, tgt)
            if best:
                out.append(best)
    out.sort(key=lambda x: -(x[0] / (x[1] + 2)))
    return out[:max_macros]


def macro_lines(b, node, tgt):
    paths = {}
    for e in range(6):
        pre = ["T"] if node.act[e] else []
        mv = ring_path(b.L, int(node.pos[e]), tgt[e]) if e in tgt else []
        paths[e] = pre + mv
    n = max(len(p) for p in paths.values())
    lines = []
    for t in range(n):
        lines.append([paths[e][t] if t < len(paths[e]) else "-" for e in range(6)])
    lines.append(["T" if e in tgt else "-" for e in range(6)])
    return lines


MACRO_BONUS = 0


def to_k(line):
    return int(sum(SYM.index(ch) * 4 ** (5 - e) for e, ch in enumerate(line)))


def beam_plan_macro(r, pattern, count, weights, width, topk, stall_limit, seed, min_cells=3, max_macros=4, macro_nodes=2):
    np.random.seed(seed)
    b = opt2.Board2(r, pattern)
    pf = positions_for(b)
    g = b.g
    nc = len(b.cells)
    root = beam.Node()
    root.pos = np.array(g.start, dtype=np.int64)
    root.act = np.zeros(6, dtype=np.int64)
    root.built = np.zeros(nc + 1, dtype=bool)
    root.done, root.prog_parent, root.line, root.progress, root.value, root.depth = 0, None, None, 0, 0.0, 0
    wg = weights[0]

    def step(node, k):
        line = [SYM[o] for o in OPTS[k]]
        ch = beam.Node()
        ch.pos, ch.act, ch.built = node.pos.copy(), node.act.copy(), node.built.copy()
        before = int(ch.built[:nc][b.P].sum())
        sub = opt2.apply2(b, ch.pos, ch.act, ch.built, line)
        after = int(ch.built[:nc][b.P].sum())
        ch.done = node.done + sub
        ch.prog_parent, ch.line = node, OPTS[k]
        ch.progress = node.progress + (b.pcount - before if sub else after - before)
        ch.depth = node.depth + 1
        return ch

    beam_ = [root]
    future = defaultdict(list)
    best_prog, stall = 0, 0
    limit = 40 * b.pcount * count + 1000
    while True:
        depth = beam_[0].depth
        if depth > limit:
            return None
        if stall >= stall_limit:
            node = max(beam_, key=lambda n: n.progress)
            bset = {b.cells[i] for i in np.nonzero(node.built[:-1])[0]}
            res = l6.next_cell(g, list(map(int, node.pos)), list(map(int, node.act)), opt2.b_P(b), bset)
            if res is None:
                return None
            for line in res[0]:
                node = step(node, to_k(line))
                if node.done >= count:
                    return beam.program(node)
            node.value = node.progress * wg
            beam_, stall, best_prog = [node], 0, node.progress
            future = defaultdict(list)
            continue
        # macro children of the best few nodes, landing at later depths
        for node in sorted(beam_, key=lambda n: -n.value)[:macro_nodes]:
            for gain, cost, tgt in chord_macros(b, pf, node, min_cells, max_macros):
                m = node
                for line in macro_lines(b, node, tgt):
                    m = step(m, to_k(line))
                    if m.done >= count:
                        break
                m.value = m.progress * wg
                future[m.depth].append(m)
        cand = {}
        for node in beam_:
            score, gain = opt2.evaluate2(b, node.pos, node.act, node.built, weights)
            top = np.argpartition(-score, topk)[:topk]
            for k in top:
                k = int(k)
                if score[k] < -1e17:
                    continue
                ch = step(node, k)
                if ch.done >= count:
                    return beam.program(ch)
                ch.value = node.progress * wg + score[k]
                key = (ch.pos.tobytes(), ch.act.tobytes(), ch.built.tobytes(), ch.done)
                if key not in cand or cand[key].value < ch.value:
                    cand[key] = ch
        for m in future.pop(depth + 1, []):
            if m.done >= count:
                return beam.program(m)
            m.value += MACRO_BONUS * wg
            key = (m.pos.tobytes(), m.act.tobytes(), m.built.tobytes(), m.done)
            if key not in cand or cand[key].value < m.value:
                cand[key] = m
        beam_ = sorted(cand.values(), key=lambda n: -n.value)[:width]
        top_prog = max(n.progress for n in beam_)
        if top_prog > best_prog:
            best_prog, stall = top_prog, 0
        else:
            stall += 1
