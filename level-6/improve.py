"""Helpers expected by campaign.py: output saving and the 12 board symmetries.

Directions (grid dy, dx): 0 E(0,2) 1 SE(1,1) 2 SW(1,-1) 3 W(0,-2) 4 NW(-1,-1) 5 NE(-1,1), clockwise.
Emitter d starts at the corner opposite its direction, so a symmetry maps start corners to start
corners. Rotation by k*60 deg maps direction d -> d+k; the mirror y -> -y maps d -> -d and turns
clockwise into counterclockwise, so F and B swap.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def save(output, programs):
    tmp = str(output) + ".tmp"
    with open(tmp, "w") as f:
        f.write("".join("\n".join(" ".join(l) for l in p) + "\n\n" for p in programs))
    os.replace(tmp, output)


def _rot(dy, dx):
    # 60 deg clockwise: E -> SE, SE -> SW
    return (dx + dy) // 2, (dx - 3 * dy) // 2


def transform(radius, pattern, symmetry):
    k, refl = symmetry % 6, symmetry // 6
    rows, cols = len(pattern), len(pattern[0])
    cy, cx = rows // 2, cols // 2
    out = [list(row) for row in pattern]
    for y in range(rows):
        for x in range(cols):
            if pattern[y][x] in "OM":
                out[y][x] = "O"
    for y in range(rows):
        for x in range(cols):
            if pattern[y][x] != "M":
                continue
            dy, dx = y - cy, x - cx
            if refl:
                dy = -dy
            for _ in range(k):
                dy, dx = _rot(dy, dx)
            out[cy + dy][cx + dx] = "M"
    # transformed emitter index for original direction d
    permutation = [((k - d) if refl else (d + k)) % 6 for d in range(6)]
    return ["".join(r) for r in out], permutation, bool(refl)


def untransform(program, permutation, reflection):
    swap = {"F": "B", "B": "F"} if reflection else {}
    return [[swap.get(line[permutation[d]], line[permutation[d]]) for d in range(6)] for line in program]
