"""Merge shard JSON results into an output file, keeping the shortest verified program per pattern.

Usage: python merge.py <input> <output> <shard.json or other out_*.txt>...
"""
import json
import os
import sys

import opt

inp, outp, shards = sys.argv[1], sys.argv[2], sys.argv[3:]
items = opt.parse(open(inp).read())
best = opt.read_progs(outp, len(items))
before = sum(len(p) for p in best)
for s in shards:
    if not os.path.exists(s):
        continue
    if s.endswith(".txt"):  # another full output file, e.g. from a different machine
        found = dict(enumerate(opt.read_progs(s, len(items))))
    else:
        found = json.load(open(s))
    for idx, prog in found.items():
        idx = int(idx)
        if prog is None:
            continue
        r, pat, count = items[idx]
        if len(prog) < len(best[idx]) and opt.l5.run(r, pat, prog) >= count:
            best[idx] = prog
for (r, pat, count), p in zip(items, best):
    assert opt.l5.run(r, pat, p) >= count
tmp = outp + ".tmp"
with open(tmp, "w") as f:
    f.write("".join("\n".join(" ".join(l) for l in p) + "\n\n" for p in best))
os.replace(tmp, outp)
print(inp, before, "->", sum(len(p) for p in best))
