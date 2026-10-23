DIRS = [(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)]


def board(r):
    n = r + 1
    rows = []
    for i in range(2 * n - 1):
        d = abs(i - (n - 1))
        k = 2 * n - 1 - d
        if d == n - 1:
            cells = ["*"] * k
        else:
            cells = ["*"] + ["O"] * (k - 2) + ["*"]
        rows.append(list("-" * d + "-".join(cells) + "-" * d))
    return rows


def ring(r):
    # border cells in clockwise order, starting at the left corner
    n = r + 1
    h = 2 * n - 1
    first = lambda i: abs(i - (n - 1))
    last = lambda i: 4 * n - 4 - abs(i - (n - 1))
    out = [(i, first(i)) for i in range(n - 1, 0, -1)]
    out += [(0, c) for c in range(first(0), last(0), 2)]
    out += [(i, last(i)) for i in range(0, h - 1)]
    out += [(h - 1, c) for c in range(last(h - 1), first(h - 1), -2)]
    out += [(i, first(i)) for i in range(h - 1, n - 1, -1)]
    return out


def run(r, prog):
    n = r + 1
    grid = board(r)
    rg = ring(r)
    idx = {p: i for i, p in enumerate(rg)}
    corners = [(n - 1, 0), (0, n - 1), (0, 3 * n - 3), (n - 1, 4 * n - 4), (2 * n - 2, 3 * n - 3), (2 * n - 2, n - 1)]
    pos = [idx[c] for c in corners]
    active = [False] * 6
    L = len(rg)
    for line in prog:
        for e, ins in enumerate(line):
            if ins == "F":
                pos[e] = (pos[e] + 1) % L
            elif ins == "B":
                pos[e] = (pos[e] - 1) % L
            elif ins == "T":
                active[e] = not active[e]
        hits = {}
        for e in range(6):
            if not active[e]:
                continue
            dr, dc = DIRS[e]
            y, x = rg[pos[e]]
            y, x = y + dr, x + dc
            while 0 <= y < len(grid) and 0 <= x < len(grid[y]):
                if grid[y][x] in "OM":
                    hits[y, x] = hits.get((y, x), 0) + 1
                y, x = y + dr, x + dc
        for (y, x), k in hits.items():
            if k == 2:
                grid[y][x] = "M"
            elif k >= 3:
                grid[y][x] = "O"
    return grid


def solve(text: str) -> str:
    lines = [l for l in text.split("\n") if l.strip()]
    n = int(lines[0])
    i = 1
    out = []
    for _ in range(n):
        r, cnt = int(lines[i]), int(lines[i + 1])
        prog = [lines[j].split() for j in range(i + 2, i + 2 + cnt)]
        i += 2 + cnt
        grid = run(r, prog)
        out.append("\n".join("".join(g) for g in grid) + "\n\n")
    return "".join(out)
