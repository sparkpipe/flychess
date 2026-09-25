"""STRETCH TRAINING (operator: "train for it — checkpoints are a tiny
sample of all possible high-accuracy flies").

Per family: lambda-mixed training — each batch row is drawn from the
own family with prob (1-lambda), from the k nearest neighbor families
with prob lambda (uniform over them). Candidates = lambda sweep x
checkpoint steps; selection = dual criterion (own >= OWN_MIN, then max
cross-family probe) over the UNION of this pool and the checkpoint-
lottery pool (stretch_search records). This trains the stretch
property directly instead of hoping a trajectory passes through it.

Outputs: singles/stretch/train_records.jsonl, chosen flies (if they
beat the lottery pool) into singles/stretch/chosen2/.
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
LAMBDAS = [float(x) for x in
           os.environ.get("LAMBDAS", "0.15,0.30").split(",")]
CAP = 3000
CK_EVERY = 250
LR = 3e-4
K_NEAR = 5
os.makedirs(f"{OUT}/chosen2", exist_ok=True)


def load_fam_deltas():
    recs = [json.loads(l) for l in open(f"{DB}/records.jsonl")]
    ids = [r["lump"] for r in recs if r.get("status") == "ok"]
    indptr = [0]
    idx_all, val_all = [], []
    for li in ids:
        z = np.load(f"{DB}/vec_{li}.npz")
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
    return ids, S


def main():
    flyfeat_cb.feat_vec(chess.Board())
    state = json.load(open(f"{CLOS}/models_2.json"))
    ids, S = load_fam_deltas()
    pos = {li: i for i, li in enumerate(ids)}
    pool_cache = {}

    def row_for(pool, fen):
        if pool not in pool_cache:
            pool_cache[pool] = {r["fen"]: r
                                for r in fc.load_pools([pool])}
        return pool_cache[pool][fen]

    rows_of, hold_of = {}, {}
    for li in ids:
        qs = [tuple(q) for q in state[li]["qs"]]
        rows = [row_for(pl, fn) for pl, fn in qs]
        rng = random.Random(1000 + li)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        cut = max(1, int(0.8 * len(rows)))
        rows_of[li] = [rows[i] for i in idx[:cut]]
        hold_of[li] = [rows[i] for i in idx[cut:]]

    rngp = random.Random(9)      # same probes as stretch_search
    all_rows = [r for li in ids for r in rows_of[li]]
    probes = {}
    near_of = {}
    near_w = {}
    TAU = float(os.environ.get("TAU", "0.10"))
    for li in ids:
        i = pos[li]
        sims = S[i].copy()
        sims[i] = -2
        near = [ids[j] for j in np.argsort(-sims)[:K_NEAR]]
        near_of[li] = near
        # per-pair lambda (operator 2026-09-26): neighbor mixing
        # weight follows measured fly-delta closeness — softmax over
        # the neighbor cosines at temperature TAU
        cs = np.array([S[i, pos[j]] for j in near])
        w = np.exp((cs - cs.max()) / TAU)
        near_w[li] = w / w.sum()
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
    recf = f"{OUT}/train_records.jsonl"
    t0 = time.time()
    for li in ids:
        own_rows = rows_of[li]
        near_list = near_of[li]
        near_rows = [r for nj in near_list
                     for r in rows_of[nj]]
        w_ij = near_w[li]
        cands = []
        for lam in LAMBDAS:
            model = tp.build_model(0)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            trng = random.Random(5000 + li * 7 + int(lam * 100))
            step = 0
            while step < CAP:
                # lambda-mixed batch: B=8 rows, own w.p. 1-lambda
                batch = []
                for _ in range(8):
                    if (not near_rows or trng.random() >= lam) \
                            and own_rows:
                        batch.append(trng.choice(own_rows))
                    elif near_rows:
                        # per-pair lambda: pick neighbor family by
                        # closeness weight, then a row from it
                        j = trng.choices(near_list, weights=w_ij)[0]
                        batch.append(trng.choice(rows_of[j]))
                    elif own_rows:
                        batch.append(trng.choice(own_rows))
                # manual tb_step on the explicit batch (B already set
                # by env, but build_tb_batch samples rows itself; so
                # train via per-row steps proportionally)
                for r in batch:
                    fc.tb_step(model, opt, [r], trng)
                step += 1
                if step % CK_EVERY == 0:
                    own, _ = fc.gate_tb(model, hold_of[li],
                                        random.Random(777),
                                        exhaustive=True)
                    prb, _ = fc.gate_tb(model, probes[li],
                                        random.Random(777),
                                        exhaustive=True)
                    cands.append({"lam": lam, "step": step,
                                  "own": round(own, 4),
                                  "probe": round(prb, 4),
                                  "sd": {k: v.detach().cpu().clone()
                                         for k, v in
                                         model.state_dict().items()}})
        ok = [c for c in cands if c["own"] >= OWN_MIN]
        chosen = max(ok or cands, key=lambda c: c["probe"]) \
            if cands else None
        if chosen is not None:
            model = tp.build_model(0)
            model.load_state_dict(chosen["sd"])
            names, flats = [], []
            for n_, p in model.named_parameters():
                names.append(n_)
                flats.append(p.detach().cpu().flatten())
            d = torch.cat(flats) - torch.cat(
                [base_sd[n_].flatten() for n_ in names])
            v, idxk = torch.topk(d.abs(), min(65536, d.numel()))
            np.savez(f"{OUT}/chosen2/vec_{li}.npz",
                     idx=idxk.numpy().astype(np.int64),
                     val=d[idxk].numpy().astype(np.float32),
                     names=np.array(names))
        rec = {"lump": li, "n_cands": len(cands),
               "chosen_own": chosen["own"] if chosen else None,
               "chosen_probe": chosen["probe"] if chosen else None,
               "lam": chosen["lam"] if chosen else None,
               "step": chosen["step"] if chosen else None,
               "frontier": sorted(
                   [{"lam": c["lam"], "st": c["step"],
                     "own": c["own"], "pr": c["probe"]}
                    for c in cands], key=lambda c: -c["pr"])[:6],
               "elapsed_s": round(time.time() - t0)}
        with open(recf, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    print("STRETCH-TRAIN-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
