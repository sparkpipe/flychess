"""CORRECTED FAIL->PASS (after the lambda-stretch negative): coverage
is a data problem — close the gap by ADDING the gap position and its
nearest corpus neighbors to family A's training mix.

Same demo position as failpass.json (pos 742). Train flyA on:
  A-train (80%) + gap position + its 15 nearest corpus positions
Checkpoint selection: own holdout >= 0.93 AND gap passing, earliest
step. Report before/after.
"""
import sys
import os
import json
import random

os.environ.setdefault("ANORM", "1")
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

DB = "/home/spec/chess-lab/singles/distbank"
CLOS = "/home/spec/chess-lab/singles/closure"
WIT = "/home/spec/chess-lab/singles/witness"
CAP = 2000
LR = 3e-4
CK = 250


def main():
    flyfeat_cb.feat_vec(chess.Board())
    demo = json.load(open("/home/spec/chess-lab/singles/stretch/"
                          "failpass.json"))
    A = demo["A"]
    meta = json.load(open(f"{WIT}/meta.json"))
    fens, pools = meta["fens"], meta["pool"]
    row_of = {f: i for i, f in enumerate(fens)}
    state = json.load(open(f"{CLOS}/models_2.json"))

    pool_cache = {}

    def row_for(pool, fen):
        if pool not in pool_cache:
            pool_cache[pool] = {r["fen"]: r
                                for r in fc.load_pools([pool])}
        return pool_cache[pool][fen]

    rows = [row_for(pl, f) for pl, f in state[A]["qs"]]
    rng = random.Random(1000 + A)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = max(1, int(0.8 * len(rows)))
    trainA = [rows[i] for i in idx[:cut]]
    holdA = [rows[i] for i in idx[cut:]]

    gi = demo["pos"]
    gap_row = row_for(pools[gi], fens[gi])
    # nearest corpus positions to the gap (exclude A's own members)
    X = np.load(f"{WIT}/pos_x.npy").astype(np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    q = Xn[gi]
    fam_of = {}
    for li, m in enumerate(state):
        for pl, f in m["qs"]:
            if f in row_of:
                fam_of[row_of[f]] = li
    sims = Xn @ q
    order = np.argsort(-sims)
    feed = [gap_row]
    k = 0
    for j in order:
        if fam_of.get(int(j)) == A:
            continue
        feed.append(row_for(pools[int(j)], fens[int(j)]))
        k += 1
        if k >= 15:
            break
    mix = trainA + feed          # gap data ENTERS the distribution

    model = tp.build_model(0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tr = random.Random(5)
    best = None
    step = 0
    while step < CAP:
        for _ in range(25):
            for _ in range(8):
                fc.tb_step(model, opt, [tr.choice(mix)], tr)
            step += 1
        if step % CK == 0:
            own, _, _ = fc.gate_tb(model, holdA, random.Random(777),
                                   exhaustive=True)
            gap, _, _ = fc.gate_tb(model, [gap_row],
                                   random.Random(777),
                                   exhaustive=True)
            rec = {"step": step, "own": round(own, 3),
                   "gap": round(gap, 3)}
            print(json.dumps(rec), flush=True)
            if gap >= 0.98 and own >= 0.93:
                if best is None:
                    best = {"step": step,
                            "sd": {k2: v.detach().cpu().clone()
                                   for k2, v in
                                   model.state_dict().items()}}
    out = {**{k: demo[k] for k in ("fen", "flyA", "flyB", "base")},
           "gapfed_own": rec["own"], "gapfed_gap": rec["gap"],
           "solved_at_step": best["step"] if best else None}
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open("/home/spec/chess-lab/singles/stretch/"
                        "gapfeed.json", "w"), indent=1)
    print("GAPFEED-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
