"""Beam search with full-board hit counts (opt2.evaluate2): allows whole-line (antiparallel) moves."""
import numpy as np

import beam
import opt
import opt2
from opt import OPTS, SYM, l6


def beam_plan2(r, pattern, count, weights, width, topk, stall_limit, seed, look=0.0, look_pool=0, init=None, limit=None):
    np.random.seed(seed)
    b = opt2.Board2(r, pattern)
    g = b.g
    nc = len(b.cells)
    root = beam.Node()
    root.pos = np.array(g.start, dtype=np.int64)
    root.act = np.zeros(6, dtype=np.int64)
    root.built = np.zeros(nc + 1, dtype=bool)
    root.done, root.prog_parent, root.line, root.progress, root.value, root.depth = 0, None, None, 0, 0.0, 0
    root.junk = 0
    if init is not None:
        # start from a given state (pos, act, full-board matter grid, submissions done)
        pos0, act0, grid0, done0 = init
        root.pos, root.act, root.done = pos0.copy(), act0.copy(), done0
        root.built[:nc] = grid0 & (b.P | b.F)
        root.junk = int(root.built[:nc][b.F].sum())
    beam_ = [root]
    best_prog, stall = 0, 0
    if limit is None:
        limit = 40 * b.pcount * count + 1000
    wg = weights[0]
    wf = weights[5] if len(weights) > 5 else 0

    def step(node, k):
        line = [SYM[o] for o in OPTS[k]]
        ch = beam.Node()
        ch.pos, ch.act, ch.built = node.pos.copy(), node.act.copy(), node.built.copy()
        before = int(ch.built[:nc][b.P].sum())
        done = node.done + opt2.apply2(b, ch.pos, ch.act, ch.built, line)
        after = int(ch.built[:nc][b.P].sum())
        ch.done = done
        ch.prog_parent, ch.line = node, OPTS[k]
        ch.progress = node.progress + (b.pcount - before if done > node.done else after - before)
        ch.depth = node.depth + 1
        ch.junk = int(ch.built[:nc][b.F].sum())
        return ch

    while True:
        if beam_[0].depth > limit:
            return None
        if stall >= stall_limit:
            node = max(beam_, key=lambda n: n.progress)
            bset = {b.cells[i] for i in np.nonzero(node.built[:-1])[0]}
            res = l6.next_cell(g, list(map(int, node.pos)), list(map(int, node.act)), opt2.b_P(b), bset)
            if res is None:
                return None
            for line in res[0]:
                k = int(sum(SYM.index(ch) * 4 ** (5 - e) for e, ch in enumerate(line)))
                node = step(node, k)
                if node.done >= count:
                    return beam.program(node)
            node.value = node.progress * wg
            beam_, stall, best_prog = [node], 0, node.progress
            continue
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
                ch.value = node.progress * wg + score[k] - wf * wg * getattr(node, "junk", 0)
                key = (ch.pos.tobytes(), ch.act.tobytes(), ch.built.tobytes(), ch.done)
                if key not in cand or cand[key].value < ch.value:
                    cand[key] = ch
        pool = sorted(cand.values(), key=lambda n: -n.value)
        if look:
            # one-step lookahead: rescore the best candidates by the best gain reachable next tick
            pool = pool[: look_pool or 4 * width]
            for n in pool:
                sc, gn = opt2.evaluate2(b, n.pos, n.act, n.built, weights)
                ok = sc > -1e17
                n.value = n.progress * wg + look * wg * (gn[ok].max() if ok.any() else 0) + (sc[ok].max() if ok.any() else 0) * 0.001
            pool.sort(key=lambda n: -n.value)
        beam_ = pool[:width]
        top_prog = max(n.progress for n in beam_)
        if top_prog > best_prog:
            best_prog, stall = top_prog, 0
        else:
            stall += 1
