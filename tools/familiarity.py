"""Familiarity signal (operator question): can a fly tell how DIFFERENT
a probe position is from what it was trained on?

For each of the 10 single-question flies: settle activations on its own
training position, then on 30 probes (the 10 questions + 20 held-out
positions from untouched chapters). Rank probes by activation-space
distance to the training position. If own-question ranks #1, the fly
carries a usable familiarity/distance signal for MoF routing — even
though margin cannot signal correctness (AUC 0.502, earlier finding).
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

heldout = []
rng = random.Random(42)
for ch in (2, 3, 6, 11, 13):
    rows = fc.load_pools([f"DEGM_Ch{ch}"])
    heldout += rng.sample(rows, 4)
probes = [chess.Board(r["fen"]) for r in heldout]
probe_tags = ["held"] * len(probes)
own_boards = []
for qi in range(N):
    e = json.loads(open(
        f"/home/spec/chess-lab/tbpools/TENQ_{qi}.jsonl").readline())
    own_boards.append(chess.Board(e["fen"]))
    probes.append(own_boards[-1])
    probe_tags.append(f"q{qi}")

fv = np.stack([flyfeat_cb.feat_vec(b)[0] for b in probes])

res = []
for qi in range(N):
    m = tp.build_model(0)
    d = torch.load(f"{OUT}/vec_{qi}.pt", map_location=DEV)
    with torch.no_grad():
        for k, p in m.named_parameters():
            p.add_(d[k].to(DEV))
        a_train = m.propagate(
            flyfeat_cb.feat_vec(own_boards[qi])[0][None, ...])
        a = m.propagate(fv)                    # (neurons, B)
        dist = (a - a_train).pow(2).sum(0).sqrt().cpu()   # (B,)
    order = torch.argsort(dist)
    rank_own = int((order == len(heldout) + qi).nonzero()[0])
    res.append({"fly": qi, "rank_own_by_dist": rank_own,
                "d_own": round(float(dist[len(heldout) + qi]), 4),
                "d_held_min": round(float(
                    dist[:len(heldout)].min()), 4),
                "nearest": probe_tags[int(order[0])]})

top1 = sum(1 for r in res if r["rank_own_by_dist"] == 0)
print(json.dumps(res, indent=1))
print(json.dumps({"own_top1": top1, "of": N,
                  "mean_rank": sum(r["rank_own_by_dist"]
                                   for r in res) / N}))
