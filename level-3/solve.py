DIRS = {"1": (0, 2), "2": (1, 1), "3": (1, -1), "4": (0, -2), "5": (-1, -1), "6": (-1, 1)}


def fire(grid):
    hits = {}
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch in DIRS:
                dr, dc = DIRS[ch]
                y, x = r + dr, c + dc
                while 0 <= y < len(grid) and 0 <= x < len(grid[y]):
                    if grid[y][x] in "OM":
                        hits[y, x] = hits.get((y, x), 0) + 1
                    y, x = y + dr, x + dc
    for (y, x), k in hits.items():
        if k == 2:
            grid[y][x] = "M"
        elif k >= 3:
            grid[y][x] = "O"


def solve(text: str) -> str:
    lines = text.split("\n")
    n = int(lines[0])
    i = 1
    out = []
    for _ in range(n):
        while not lines[i].strip():
            i += 1
        _, rows, _ = map(int, lines[i].split())
        grid = [list(l.strip()) for l in lines[i + 1:i + 1 + rows]]
        i += 1 + rows
        fire(grid)
        out.append("\n".join("".join(g) for g in grid) + "\n\n")
    return "".join(out)
