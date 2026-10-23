DIRS = {"1": (0, 2), "2": (1, 1), "3": (1, -1), "4": (0, -2), "5": (-1, -1), "6": (-1, 1)}


def fire(grid):
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch in DIRS:
                dr, dc = DIRS[ch]
                y, x = r + dr, c + dc
                while 0 <= y < len(grid) and 0 <= x < len(grid[y]):
                    if grid[y][x] == "O":
                        grid[y][x] = "M"
                    y, x = y + dr, x + dc


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
