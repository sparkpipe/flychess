"""DECISIVE TEST: does DISTRIBUTION training give local generalization?

Take one coherent lump family (closure model #3, 259 positions), split
80/20 train/held-out. Train a FRESH fly on the 80% with interleaved
batches (B=8 rows/step — distribution protocol, NOT single-position),
exhaustive-gate BOTH sets. Verdict:
  held-out ~= train  -> substrate generalizes under distribution
                        training -> fly factories train distributions
  held-out << train  -> no in-distribution generalization either ->
                        go Leela-style direct W_sens training
"""
import sys
import os
import json
import random
import time

os.environ.setdefault("ANORM", "1")
os.environ.setdefault("B", "8")            # distribution protocol
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

CLOS = "/home/spec/chess-lab/singles/closure"
CAP = int(os.environ.get("CAP", "3000"))
LR = 3e-4
GATE_EVERY = 50
MODEL_IDX = int(os.environ.get("LUMP", "3"))


def main():
    flyfeat_cb.feat_vec(chess.Board())
    state = json.load(open(f"{CLOS}/models_2.json"))
    qs = [tuple(q) for q in state[MODEL_IDX]["qs"]]
    print(json.dumps({"lump": MODEL_IDX, "positions": len(qs)}),
          flush=True)
    pool_cache = {}

    def row_for(pool, fen):
        if pool not in pool_cache:
            pool_cache[pool] = {r["fen"]: r
                                for r in fc.load_pools([pool])}
        return pool_cache[pool][fen]

    rows = [row_for(pl, fn) for pl, fn in qs]
    rng = random.Random(42)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = int(0.8 * len(rows))
    train_rows = [rows[i] for i in idx[:cut]]
    hold_rows = [rows[i] for i in idx[cut:]]
    print(json.dumps({"train": len(train_rows),
                      "holdout": len(hold_rows)}), flush=True)

    m = tp.build_model(0)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    trng = random.Random(7)

    def gate(rrs):
        pair, _, _ = fc.gate_tb(m, rrs, random.Random(777),
                                exhaustive=True)
        return pair

    print(json.dumps({"step0_train": round(gate(train_rows), 3),
                      "step0_hold": round(gate(hold_rows), 3)}),
          flush=True)
    t0 = time.time()
    step = 0
    while step < CAP:
        for _ in range(GATE_EVERY):
            fc.tb_step(m, opt, train_rows, trng)   # B=8 mixed rows
            step += 1
        print(json.dumps({"step": step,
                          "train": round(gate(train_rows), 3),
                          "hold": round(gate(hold_rows), 3),
                          "elapsed_s": round(time.time() - t0)}),
              flush=True)
    print("DIST-FLY-TEST-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
