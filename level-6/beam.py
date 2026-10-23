"""Level 6 optimizer v3: beam search over ticks using the crossing-based evaluator from opt.py.

Usage: python beam.py <input> <output> <width> <topk> [config indices]
"""
import os
import sys

import numpy as np

import opt
from opt import OPTS, SYM, Board, evaluate, l5, l6


class Node:
    __slots__ = ("pos", "act", "built", "done", "prog_parent", "line", "progress", "value", "depth", "junk")


def child_of(b, node, k, gain_cells):
    c = Node()
    line = OPTS[k]
    pos = node.pos.copy()
    act = node.act.copy()
    for e, o in enumerate(line):
        if o == 1:
            pos[e] = (pos[e] + 1) % b.L
        elif o == 2:
            pos[e] = (pos[e] - 1) % b.L
        elif o == 3:
            act[e] ^= 1
    built = node.built.copy()
    built[gain_cells] = True
    done = node.done
    if built[:-1][b.pmask[:-1]].all():
        built[:] = False
        done += 1
    c.pos, c.act, c.built, c.done = pos, act, built, done
    c.prog_parent, c.line = node, line
    c.progress = node.progress + len(gain_cells)
    c.depth = node.depth + 1
    return c


def program(node):
    lines = []
    while node.prog_parent is not None:
        if isinstance(node.line, list):
            lines.extend(reversed(node.line))
        else:
            lines.append([SYM[o] for o in node.line])
        node = node.prog_parent
    return lines[::-1]


def beam_plan(r, pattern, count, weights, width, topk, stall_limit, seed):
    np.random.seed(seed)
    b = Board(r, pattern)
    g = b.g
    root = Node()
    root.pos = np.array(g.start, dtype=np.int64)
    root.act = np.zeros(6, dtype=np.int64)
    root.built = np.zeros(len(b.cells) + 1, dtype=bool)
    root.done, root.prog_parent, root.line, root.progress, root.value, root.depth = 0, None, None, 0, 0.0, 0
    beam = [root]
    best_prog, stall = 0, 0
    limit = 40 * b.pcount * count + 1000
    wg = weights[0]
    while True:
        if beam[0].depth > limit:
            return None
        if stall >= stall_limit:
            node = max(beam, key=lambda n: n.progress)
            bset = {b.cells[i] for i in np.nonzero(node.built[:-1])[0]}
            res = l6.next_cell(g, list(map(int, node.pos)), list(map(int, node.act)), b.P, bset)
            if res is None:
                return None
            # replay fallback lines one by one to keep state exact
            for line in res[0]:
                k = int(sum(SYM.index(ch) * 4 ** (5 - e) for e, ch in enumerate(line)))
                score, gain, pc, H = evaluate(b, node.pos, node.act, node.built, weights)
                newp = pc[k][(H[k] == 2) & (b.status[pc[k]] == 2) & ~node.built[pc[k]] & (pc[k] >= 0)]
                node = child_of(b, node, k, np.unique(newp))
                if node.done >= count:
                    return program(node)
            node.value = node.progress * wg
            beam = [node]
            stall = 0
            best_prog = node.progress
            continue
        cand = {}
        for node in beam:
            score, gain, pc, H = evaluate(b, node.pos, node.act, node.built, weights)
            top = np.argpartition(-score, topk)[:topk]
            for k in top:
                k = int(k)
                if score[k] < -1e17:
                    continue
                m = (H[k] == 2) & (b.status[pc[k]] == 2) & ~node.built[pc[k]] & (pc[k] >= 0)
                ch = child_of(b, node, k, np.unique(pc[k][m]))
                if ch.done >= count:
                    return program(ch)
                ch.value = node.progress * wg + score[k]
                key = (ch.pos.tobytes(), ch.act.tobytes(), ch.built.tobytes(), ch.done)
                if key not in cand or cand[key].value < ch.value:
                    cand[key] = ch
        beam = sorted(cand.values(), key=lambda n: -n.value)[:width]
        top_prog = max(n.progress for n in beam)
        if top_prog > best_prog:
            best_prog, stall = top_prog, 0
        else:
            stall += 1


CONFIGS = [((1000, 30, 1), 6), ((1000, 15, 0.5), 6), ((1000, 60, 2), 8), ((1000, 30, 1), 10)]


def main():
    inp, outp, width, topk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    cfg = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else range(len(CONFIGS))
    items = opt.parse(open(inp).read())
    best = opt.read_progs(outp, len(items))
    for idx, (r, pat, count) in enumerate(items):
        if best[idx] is None or l5.run(r, pat, best[idx]) < count:
            best[idx] = l6.plan(r, pat, count)
        for ci in cfg:
            weights, stall = CONFIGS[ci]
            prog = beam_plan(r, pat, count, weights, width, topk, stall, seed=ci * 7919 + idx)
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
    print(inp, "score", sum(len(p) for p in best), flush=True)


if __name__ == "__main__":
    main()
