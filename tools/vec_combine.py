"""Combination experiments on the single-solution vectors (operator
directive: 'then we can experiment with various combinations').

All flies share the seed-0 init, so deltas are additive by construction.
Tests:
  1. merged-all: init + sum(delta_i, solved flies) -> gate each
     constituent question on the merged model.
  2. pairwise: init + vec_a + vec_b for the most-similar and
     most-orthogonal pairs by top-1000-support cosine -> gate both.
"""
import sys
import os
import json
import random
import chess
import torch

sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

flyfeat_cb.feat_vec(chess.Board())
DEV = tp.DEV
OUT = "/home/spec/chess-lab/ten_q"

report = json.load(open(f"{OUT}/report.json"))
N = len(report)
solved = [i for i, r in enumerate(report)
          if r["stable_step"] is not None and i != 7]

vecs = {i: torch.load(f"{OUT}/vec_{i}.pt", map_location=DEV)
        for i in range(N)}
rows = {i: fc.load_pools([f"TENQ_{i}"]) for i in range(N)}


def merged_model(idxs):
    m = tp.build_model(0)
    with torch.no_grad():
        for i in idxs:
            for k, p in m.named_parameters():
                p.add_(vecs[i][k].to(DEV))
    return m


def gate(m, qi):
    pair, _, _ = fc.gate_tb(m, rows[qi], random.Random(777),
                            exhaustive=True)
    return round(pair, 3)


# support-restricted cosine for pair selection (top-1000)
def flat(d):
    return torch.cat([v.flatten() for v in d.values()])


def topk(d, k=1000):
    return torch.topk(flat(d).abs(), k).indices


sups = {i: set(topk(vecs[i]).tolist()) for i in range(N)}
pairs = []
for a in range(N):
    for b in range(a + 1, N):
        u = sorted(sups[a] | sups[b])
        va = flat(vecs[a])[u]
        vb = flat(vecs[b])[u]
        cos = float(torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-12))
        pairs.append((cos, a, b))
pairs.sort()
most_orth = pairs[0]
most_sim = max(pairs)

print(json.dumps({"solved_flies": solved}), flush=True)

m = merged_model(solved)
res = {qi: gate(m, qi) for qi in solved}
print(json.dumps({"MERGED-ALL": res,
                  "all_pass": all(v >= 0.98 for v in res.values())}),
      flush=True)

for tag, (cos, a, b) in (("MOST-SIMILAR", most_sim),
                         ("MOST-ORTHOGONAL", most_orth)):
    mm = merged_model([a, b])
    print(json.dumps({tag: {"pair": [a, b], "support_cos":
                            round(cos, 3),
                           "gates": {a: gate(mm, a),
                                     b: gate(mm, b)}}}), flush=True)

# every pair among solved flies, both gated — the full merge map
grid = {}
for a in solved:
    for b in solved:
        if a < b:
            mm = merged_model([a, b])
            ga, gb = gate(mm, a), gate(mm, b)
            grid[f"{a}+{b}"] = [ga, gb]
print(json.dumps({"PAIRWISE-GRID": grid}), flush=True)
