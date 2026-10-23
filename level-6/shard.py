"""Run beam search on one shard of an input; write {idx: program} JSON.

Usage: python shard.py <input> <shard_out.json> <i> <n> <width> <topk> <config>
"""
import json
import sys

import beam
import beam2
import opt

inp, out, i, n, W, K, ci = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:8])
items = opt.parse(open(inp).read())
# config >= 100 selects the full-board beam (beam2) with BEAM2_CONFIGS[ci - 100]
# config 102: full-board beam with one-step lookahead rescoring
BEAM2_CONFIGS = [((1000, 30, 0, 0), 6), ((1000, 30, 0, 0, 1.0), 6), ((1000, 30, 0, 0), 6)]
if ci >= 100:
    weights, stall = BEAM2_CONFIGS[ci - 100]
    planner = beam2.beam_plan2
    if ci == 102:
        def planner(*a, seed):
            return beam2.beam_plan2(*a, seed=seed, look=1.0, look_pool=4 * W)
else:
    weights, stall = beam.CONFIGS[ci]
    planner = beam.beam_plan
res = {}
for idx, (r, pat, count) in enumerate(items):
    if idx % n != i:
        continue
    prog = planner(r, pat, count, weights, W, K, stall, seed=ci * 7919 + idx)
    if prog is not None and opt.l5.run(r, pat, prog) >= count:
        res[idx] = prog
    json.dump(res, open(out, "w"))
