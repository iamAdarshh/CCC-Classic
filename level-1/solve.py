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
        rows.append("-" * d + "-".join(cells) + "-" * d)
    return "\n".join(rows)


def solve(text: str) -> str:
    nums = list(map(int, text.split()))
    return "".join(board(r) + "\n\n" for r in nums[1:1 + nums[0]])
