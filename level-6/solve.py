import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "level-5"))
from solve import DIRS, board, ring, run as simulate  # noqa: E402

MOVES = [("-", 0, 0), ("F", 1, 0), ("B", -1, 0), ("T", 0, 1)]


def line_key(e, y, x):
    return (y, x - y, x + y)[e % 3]


class Geometry:
    def __init__(self, r):
        n = r + 1
        self.grid = board(r)
        self.rg = ring(r)
        self.L = len(self.rg)
        idx = {p: i for i, p in enumerate(self.rg)}
        corners = [(n - 1, 0), (0, n - 1), (0, 3 * n - 3), (n - 1, 4 * n - 4), (2 * n - 2, 3 * n - 3), (2 * n - 2, n - 1)]
        self.start = [idx[c] for c in corners]
        interior = [(y, x) for y, row in enumerate(self.grid) for x, ch in enumerate(row) if ch == "O"]
        inside = set(interior)
        # key[e][p]: line key of the beam of emitter e at ring position p, None if it misses the board
        self.key = []
        for e in range(6):
            dy, dx = DIRS[e]
            ks = []
            for (y, x) in self.rg:
                ks.append(line_key(e, y, x) if (y + dy, x + dx) in inside else None)
            self.key.append(ks)
        # cross[(c1, k1, c2, k2)] -> cell, for classes c1 < c2
        self.cross = {}
        for (y, x) in interior:
            ks = (y, x - y, x + y)
            for c1 in range(3):
                for c2 in range(c1 + 1, 3):
                    self.cross[c1, ks[c1], c2, ks[c2]] = (y, x)

    def meet(self, e1, p1, e2, p2):
        k1, k2 = self.key[e1][p1], self.key[e2][p2]
        if k1 is None or k2 is None:
            return None
        c1, c2 = e1 % 3, e2 % 3
        if c1 > c2:
            c1, c2, k1, k2 = c2, c1, k2, k1
        return self.cross.get((c1, k1, c2, k2))


PAIRS = [(i, j) for i in range(6) for j in range(i + 1, 6) if i % 3 != j % 3]


def next_cell(g, pos, act, P, built):
    """BFS over all emitter pairs at once; returns list of instruction lines reaching a new pattern cell."""
    L = g.L
    frontiers = []
    for (i, j) in PAIRS:
        s = (pos[i], act[i], pos[j], act[j])
        frontiers.append(({s: None}, [s]))
    depth = 0
    while True:
        depth += 1
        if depth > 4 * L:
            return None
        for pi, (i, j) in enumerate(PAIRS):
            seen, front = frontiers[pi]
            nxt = []
            for s in front:
                p1, a1, p2, a2 = s
                for m1, d1, t1 in MOVES:
                    q1, b1 = (p1 + d1) % L, a1 ^ t1
                    for m2, d2, t2 in MOVES:
                        q2, b2 = (p2 + d2) % L, a2 ^ t2
                        ns = (q1, b1, q2, b2)
                        if ns in seen:
                            continue
                        c = g.meet(i, q1, j, q2) if b1 and b2 else None
                        if c is not None and c not in P:
                            continue
                        seen[ns] = (s, m1, m2)
                        if c is not None and c not in built:
                            return finish(pos, act, i, j, seen, ns), c
                        nxt.append(ns)
            frontiers[pi] = (seen, nxt)


def finish(pos, act, i, j, seen, s):
    steps = []
    while seen[s] is not None:
        prev, m1, m2 = seen[s]
        steps.append((m1, m2))
        s = prev
    steps.reverse()
    lines = []
    for k, (m1, m2) in enumerate(steps):
        line = ["-"] * 6
        if k == 0:
            for e in range(6):
                if e not in (i, j) and act[e]:
                    line[e] = "T"
        line[i], line[j] = m1, m2
        lines.append(line)
    return lines


def plan(r, pattern, count):
    g = Geometry(r)
    P = {(y, x) for y, row in enumerate(pattern) for x, ch in enumerate(row) if ch == "M"}
    pos = list(g.start)
    act = [0] * 6
    prog = []
    for _ in range(count):
        built = set()
        while len(built) < len(P):
            lines, c = next_cell(g, pos, act, P, built)
            for line in lines:
                for e, ins in enumerate(line):
                    if ins == "F":
                        pos[e] = (pos[e] + 1) % g.L
                    elif ins == "B":
                        pos[e] = (pos[e] - 1) % g.L
                    elif ins == "T":
                        act[e] ^= 1
                prog.append(line)
                # every tick of the path may (re)build pattern cells only
                for i in range(6):
                    for j in range(i + 1, 6):
                        if act[i] and act[j]:
                            m = g.meet(i, pos[i], j, pos[j])
                            if m is not None:
                                built.add(m)
    return prog


def solve(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    n = int(lines[0])
    i = 1
    out = []
    total = 0
    for _ in range(n):
        r, rows, _, count = map(int, lines[i].split())
        pattern = lines[i + 1:i + 1 + rows]
        i += 1 + rows
        prog = plan(r, pattern, count)
        got = simulate(r, pattern, prog)
        assert got >= count, (r, count, got)
        total += len(prog)
        out.append("\n".join(" ".join(l) for l in prog) + "\n\n")
    print("score (total lines):", total, file=sys.stderr)
    return "".join(out)
