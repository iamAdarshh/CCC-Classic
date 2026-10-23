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


def run(r, pattern, prog):
    n = r + 1
    grid = board(r)
    rg = ring(r)
    idx = {p: i for i, p in enumerate(rg)}
    corners = [(n - 1, 0), (0, n - 1), (0, 3 * n - 3), (n - 1, 4 * n - 4), (2 * n - 2, 3 * n - 3), (2 * n - 2, n - 1)]
    pos = [idx[c] for c in corners]
    active = [False] * 6
    L = len(rg)
    pcells = [(y, x) for y, row in enumerate(pattern) for x, ch in enumerate(row) if ch == "M"]
    pset = set(pcells)
    nbrs = {(y + dy, x + dx) for y, x in pcells for dy, dx in DIRS} - pset
    nbrs = [(y, x) for y, x in nbrs if 0 <= y < len(grid) and 0 <= x < len(grid[y])]
    count = 0
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
        if all(grid[y][x] == "M" for y, x in pcells) and not any(grid[y][x] == "M" for y, x in nbrs):
            for y, x in pcells:
                grid[y][x] = "O"
            count += 1
    return count


def solve(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    n = int(lines[0])
    i = 1
    out = []
    for _ in range(n):
        r, rows, _ = map(int, lines[i].split())
        pattern = lines[i + 1:i + 1 + rows]
        cnt = int(lines[i + 1 + rows])
        prog = [lines[j].split() for j in range(i + 2 + rows, i + 2 + rows + cnt)]
        i += 2 + rows + cnt
        out.append(str(run(r, pattern, prog)))
    return "\n".join(out) + "\n"
