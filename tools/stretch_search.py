"""STRETCH SEARCH (operator directive 2026-09-26): per family, search
the fly population (checkpoints x seeds) for candidates that keep
family accuracy high AND generalize furthest toward neighboring
families — smoothing the 138 into overlapping, connected radii.

Per family: 2 seeds x checkpoints every 250 steps (<=3000) of the
distribution protocol. Each candidate gated on:
  own  = family held-out 20% (same split/seed as distbank v1)
  probe = ~15 positions from each of 5 nearest families (by v1
          fly-delta cosine) + 100 random cross-family positions
Selection: own >= OWN_MIN, then max probe. Full frontier recorded.
Outputs: singles/stretch/{records.jsonl, matrix.npy, chosen/{vec,meta}}
"""
import sys
import os
import json
import random
import time

os.environ.setdefault("ANORM", "1")
os.environ.setdefault("B", "8")
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc
from scipy import sparse

CLOS = "/home/spec/chess-lab/singles/closure"
DB = "/home/spec/chess-lab/singles/distbank"
OUT = "/home/spec/chess-lab/singles/stretch"
OWN_MIN = float(os.environ.get("OWN_MIN", "0.95"))
CAP = 3000
CK_EVERY = 250
LR = 3e-4
SEEDS = (0, 1)
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/chosen", exist_ok=True)


def load_fam_deltas():
    """v1 distbank vectors -> cosine for neighbor discovery."""
    recs = [json.loads(l) for l in open(f"{DB}/records.jsonl")]
    ids = [r["lump"] for r in recs if r.get("status") == "ok"]
    mats = {}
    indptr = [0]
    idx_all, val_all = [], []
    for li in ids:
        z = np.load(f"{DB}/vec_{li}.npz")
        mats[li] = (z["idx"], z["val"])
        idx_all.append(z["idx"])
        val_all.append(z["val"])
        indptr.append(indptr[-1] + len(z["idx"]))
    A = sparse.csr_matrix(
        (np.concatenate(val_all), np.concatenate(idx_all),
         np.array(indptr)),
        shape=(len(ids), int(np.max(np.concatenate(idx_all))) + 1))
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    norms[norms < 1e-12] = 1.0
    A.data /= np.repeat(norms, np.diff(indptr))
    S = (A @ A.T).toarray()
    return ids, mats, S


def main():
    flyfeat_cb.feat_vec(chess.Board())
    state = json.load(open(f"{CLOS}/models_2.json"))
    ids, mats, S = load_fam_deltas()
    pos = {li: i for i, li in enumerate(ids)}
    pool_cache = {}

    def row_for(pool, fen):
        if pool not in pool_cache:
            pool_cache[pool] = {r["fen"]: r
                                for r in fc.load_pools([pool])}
        return pool_cache[pool][fen]

    rows_of = {}
    hold_of = {}
    for li in ids:
        qs = [tuple(q) for q in state[li]["qs"]]
        rows = [row_for(pl, fn) for pl, fn in qs]
        rng = random.Random(1000 + li)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        cut = max(1, int(0.8 * len(rows)))
        rows_of[li] = [rows[i] for i in idx[:cut]]
        hold_of[li] = [rows[i] for i in idx[cut:]]

    # probes: 5 nearest families x 15 + 100 random cross-family
    rngp = random.Random(9)
    all_rows = [r for li in ids for r in rows_of[li]]
    probes = {}
    for li in ids:
        i = pos[li]
        sims = S[i].copy()
        sims[i] = -2
        near = [ids[j] for j in np.argsort(-sims)[:5]]
        pr = []
        for nj in near:
            src = rows_of[nj]
            if src:
                pr += rngp.sample(src, min(15, len(src)))
        pr += rngp.sample(all_rows, min(100, len(all_rows)))
        probes[li] = pr

    base = tp.build_model(0)
    base_sd = {k: v.detach().cpu().clone()
               for k, v in base.named_parameters()}
    base_cat = torch.cat([base_sd[k].flatten()
                          for k, _ in base.named_parameters()])

    recf = f"{OUT}/records.jsonl"
    M = np.zeros((len(ids), len(ids)), np.float32)
    t0 = time.time()
    for li in ids:
        train_rows = rows_of[li]
        cands = []
        for seed in SEEDS:
            model = tp.build_model(0)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            trng = random.Random(2000 + 31 * seed + li)
            step = 0
            while step < CAP:
                for _ in range(25):
                    fc.tb_step(model, opt, train_rows, trng)
                    step += 1
                if step % CK_EVERY == 0:
                    own, _ = fc.gate_tb(model, hold_of[li],
                                        random.Random(777),
                                        exhaustive=True)
                    probe, _ = fc.gate_tb(model, probes[li],
                                           random.Random(777),
                                           exhaustive=True)
                    cands.append({"seed": seed, "step": step,
                                  "own": round(own, 4),
                                  "probe": round(probe, 4),
                                  "sd": {k: v.detach().cpu().clone()
                                         for k, v in
                                         model.state_dict().items()}})
        ok = [c for c in cands if c["own"] >= OWN_MIN]
        chosen = max(ok or cands, key=lambda c: c["probe"]) \
            if cands else None
        if chosen is None:
            continue
        model = tp.build_model(0)
        model.load_state_dict(chosen["sd"])
        names, flats = [], []
        for n_, p in model.named_parameters():
            names.append(n_)
            flats.append(p.detach().cpu().flatten())
        d = torch.cat(flats) - torch.cat(
            [base_sd[n_].flatten() for n_ in names])
        v, idxk = torch.topk(d.abs(), min(65536, d.numel()))
        np.savez(f"{OUT}/chosen/vec_{li}.npz",
                 idx=idxk.numpy().astype(np.int64),
                 val=d[idxk].numpy().astype(np.float32),
                 names=np.array(names))
        # stretch matrix row: chosen fly vs every family's holdout
        for lj in ids:
            if hold_of[lj]:
                sc, _ = fc.gate_tb(model, hold_of[lj],
                                   random.Random(777),
                                   exhaustive=True)
                M[pos[li], pos[lj]] = sc
        rec = {"lump": li, "n_cands": len(cands),
               "own": chosen["own"], "probe": chosen["probe"],
               "seed": chosen["seed"], "step": chosen["step"],
               "frontier": [{"s": c["seed"], "st": c["step"],
                             "own": c["own"], "pr": c["probe"]}
                            for c in sorted(
                                cands, key=lambda c: -c["probe"])[:6]],
               "elapsed_s": round(time.time() - t0)}
        with open(recf, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    np.save(f"{OUT}/matrix.npy", M)
    print("STRETCH-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
