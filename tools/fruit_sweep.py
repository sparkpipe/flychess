"""PHASE 1 — low-hanging-fruit sweep (operator directive 2026-09-20):
sweep ALL instant merges first. Greedily grow verified lumps: start
from the highest-cosine seed, try adding candidates in descending
cosine order, MERGE only when the superposed model passes exhaustive
gates over the whole union (zero repair). A lump closes after 25
consecutive failed candidates. Leftovers feed phase 2 (repair
pairwise/tree).

CPU only (FCDEV=cpu, CUDA hidden, nice). Every accepted merge is
gate-VERIFIED at merge time — lumps are exhaustively-correct models,
not predictions. Checkpoint: singles/fruit/ after each closed lump.
"""
import sys
import os
import json
import random
import time

_DEV = os.environ.get("MERGE_DEV", "cpu")
if _DEV != "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("FCDEV", _DEV)
os.environ.setdefault("ANORM", "1")
os.environ.setdefault("B", "1")
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc
from scipy import sparse

torch.set_num_threads(16)
IN = "/home/spec/chess-lab/singles"
REP = "/home/spec/chess-lab/singles_repair"
FRUIT = "/home/spec/chess-lab/singles/fruit"
GIVE_UP_AFTER = 25
LR = 3e-4
GATE_EVERY = 10
K = 4096
os.makedirs(FRUIT, exist_ok=True)


def load_vecs():
    rep = {}
    if os.path.exists(f"{REP}/records.jsonl"):
        for line in open(f"{REP}/records.jsonl"):
            r = json.loads(line)
            rep[r["fen"]] = r
    seen = set()
    vecs = []
    for line in open(f"{IN}/records.jsonl"):
        r = json.loads(line)
        fen = r["fen"]
        if fen in seen:
            continue
        solved = r["stable_step"] is not None
        if fen in rep:
            pass
        elif not solved:
            continue
        seen.add(fen)
        src = REP if fen in rep else IN
        gi = rep[fen]["gi"] if fen in rep else r["gi"]
        z = np.load(f"{src}/vec_{gi}.npz")
        vecs.append({"pool": r["pool"], "fen": fen,
                     "idx": z["idx"].astype(np.int64),
                     "val": z["val"].astype(np.float32)})
    return vecs


def cos_matrix(vecs):
    n = len(vecs)
    indptr = np.zeros(n + 1, dtype=np.int64)
    for i, m in enumerate(vecs):
        indptr[i + 1] = indptr[i] + len(m["idx"])
    idx = np.concatenate([m["idx"] for m in vecs])
    val = np.concatenate([m["val"] for m in vecs])
    A = sparse.csr_matrix((val, idx, indptr),
                          shape=(n, int(idx.max()) + 1))
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    norms[norms < 1e-12] = 1.0
    A.data /= np.repeat(norms, np.diff(indptr))
    return (A @ A.T).toarray()


def main():
    flyfeat_cb.feat_vec(chess.Board())
    vecs = load_vecs()
    n = len(vecs)
    print(json.dumps({"vectors": n}), flush=True)
    S = cos_matrix(vecs)
    np.save(f"{FRUIT}/sim.npy", S)

    base = tp.build_model(0)
    base_state = {k: p.detach().clone()
                  for k, p in base.named_parameters()}
    base_cat = torch.cat([base_state[k].flatten()
                          for k, _ in base.named_parameters()])
    pool_cache = {}

    def row_for(pool, fen):
        if pool not in pool_cache:
            pool_cache[pool] = {r["fen"]: r
                                for r in fc.load_pools([pool])}
        return pool_cache[pool][fen]

    def build(delta_idx, delta_val):
        with torch.no_grad():
            for k, p in base.named_parameters():
                p.copy_(base_state[k])
            flat = base_cat.clone()
            flat.index_add_(0, torch.from_numpy(delta_idx).to(fc.DEV),
                            torch.from_numpy(delta_val).to(fc.DEV))
            off = 0
            for _, p in base.named_parameters():
                n_ = p.numel()
                p.copy_(flat[off:off + n_].view_as(p))
                off += n_

    def gates_pass(members_idx):
        rows = [row_for(vecs[i]["pool"], vecs[i]["fen"])
                for i in members_idx]
        pair, _, _ = fc.gate_tb(base, rows, random.Random(777),
                                exhaustive=True)
        return pair >= 0.999        # every question, no hiding

    alive = set(range(n))
    lumps = []
    t0 = time.time()
    lump_k = 0
    while alive:
        # seed: alive vector with highest avg cosine to other alive
        alive_l = sorted(alive)
        sub = S[np.ix_(alive_l, alive_l)]
        avg = (sub.sum(1) - np.diag(sub)) / max(len(alive_l) - 1, 1)
        seed = alive_l[int(np.argmax(avg))]
        members = [seed]
        member_set = {seed}
        # candidate order: desc cosine to seed (approx: recompute vs
        # current members each rejection round is too slow; seed-order
        # with fail budget is the cheap heuristic)
        cand = [i for i in alive_l if i != seed]
        cand.sort(key=lambda i: -S[seed, i])
        fails = 0
        tried = set()
        # grow: aggregate delta grows by successful merges
        didx = vecs[seed]["idx"].copy()
        dval = vecs[seed]["val"].copy()
        ci = 0
        while ci < len(cand) and fails < GIVE_UP_AFTER:
            i = cand[ci]
            ci += 1
            u = np.concatenate([didx, vecs[i]["idx"]])
            v = np.concatenate([dval, vecs[i]["val"]])
            agg = np.zeros(int(u.max()) + 1, np.float32)
            np.add.at(agg, u, v)
            nzi = np.nonzero(agg)[0]
            build(nzi, agg[nzi])
            test = members + [i]
            if gates_pass(test):
                members = test
                member_set.add(i)
                didx, dval = nzi, agg[nzi]
                fails = 0
            else:
                fails += 1
                # brief repair attempt is phase 2's job; phase 1 = 0 work
        lump_k += 1
        lumps.append({"members": members,
                      "didx": didx, "dval": dval})
        alive -= member_set
        rec = {"lump": lump_k, "size": len(members), "seed": seed,
               "alive_left": len(alive),
               "elapsed_s": round(time.time() - t0)}
        print(json.dumps(rec), flush=True)
        with open(f"{FRUIT}/lumps_state.json", "w") as f:
            json.dump({"lumps": [{"size": len(l["members"]),
                                  "members": l["members"]}
                                 for l in lumps],
                       "alive": sorted(alive),
                       "vectors": n}, f)
    tot = sum(len(l["members"]) for l in lumps)
    sizes = sorted((len(l["members"]) for l in lumps), reverse=True)
    out = {"lumps": len(lumps), "covered": tot,
           "sizes_top10": sizes[:10],
           "singletons": sum(1 for s in sizes if s == 1),
           "elapsed_s": round(time.time() - t0)}
    json.dump(out, open(f"{FRUIT}/summary.json", "w"), indent=1)
    print(json.dumps(out), flush=True)
    np.savez_compressed(
        f"{FRUIT}/lump_deltas.npz",
        **{f"l{k:04d}_{sfx}": arr for k, l in enumerate(lumps)
           for sfx, arr in (("idx", l["didx"]),
                            ("val", l["dval"]))})
    with open(f"{FRUIT}/lumps.json", "w") as f:
        json.dump([{"members": [{"pool": vecs[i]["pool"],
                                 "fen": vecs[i]["fen"]}
                                for i in l["members"]]}
                   for l in lumps], f)
    print("FRUIT-SWEEP-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
