"""PHASE 2 — instant-pair closure (operator directive 2026-09-20):
after the fruit sweep, try ALL pairs of current models for instant
(zero-work) superposition passes; merge what passes; REPEAT the all-
pairs sweep until an entire iteration yields zero merges — the fixed
point of instant merging.

Per iteration:
  - cosine matrix over current model deltas (CSR)
  - scan pairs in DESCENDING cosine order; both endpoints must be
    alive; build init + da + db; batched exhaustive gate over the
    union questions; merge only on a perfect pass
  - merged model carries the union question set and summed delta
Stop when an iteration merges nothing. Checkpoint per iteration:
singles/closure/. CPU only, nice.
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
FRUIT = "/home/spec/chess-lab/singles/fruit"
CLOS = "/home/spec/chess-lab/singles/closure"
LR = 3e-4
os.makedirs(CLOS, exist_ok=True)


def load_models():
    st = json.load(open(f"{FRUIT}/lumps_state.json"))
    z = np.load(f"{FRUIT}/lump_deltas.npz")
    state = json.load(open(f"{FRUIT}/lumps.json"))
    models = []
    for k, l in enumerate(state):
        models.append({"qs": [(m["pool"], m["fen"])
                              for m in l["members"]],
                       "idx": z[f"l{k:04d}_idx"],
                       "val": z[f"l{k:04d}_val"]})
    return models


def cos_matrix(models):
    n = len(models)
    indptr = np.zeros(n + 1, dtype=np.int64)
    for i, m in enumerate(models):
        indptr[i + 1] = indptr[i] + len(m["idx"])
    idx = np.concatenate([m["idx"] for m in models])
    val = np.concatenate([m["val"] for m in models])
    A = sparse.csr_matrix((val, idx, indptr),
                          shape=(n, int(idx.max()) + 1))
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    norms[norms < 1e-12] = 1.0
    A.data /= np.repeat(norms, np.diff(indptr))
    return (A @ A.T).toarray()


def main():
    flyfeat_cb.feat_vec(chess.Board())
    models = load_models()
    print(json.dumps({"models_in": len(models)}), flush=True)
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

    def try_merge(ma, mb):
        rows = [row_for(pl, fn) for pl, fn in ma["qs"] + mb["qs"]]
        with torch.no_grad():
            for k, p in base.named_parameters():
                p.copy_(base_state[k])
            flat = base_cat.clone()
            for m in (ma, mb):
                flat.index_add_(0, torch.from_numpy(m["idx"]).to(fc.DEV),
                                torch.from_numpy(m["val"]).to(fc.DEV))
            off = 0
            for _, p in base.named_parameters():
                n_ = p.numel()
                p.copy_(flat[off:off + n_].view_as(p))
                off += n_
        pair, _, _ = fc.gate_tb(base, rows, random.Random(777),
                                exhaustive=True)
        return pair >= 0.999, rows

    it = 0
    t0 = time.time()
    while True:
        it += 1
        t_i = time.time()
        S = cos_matrix(models)
        n = len(models)
        iu = np.triu_indices(n, 1)
        order = np.argsort(-S[iu])
        alive = set(range(n))
        merged = 0
        for o in order:
            a, b = int(iu[0][o]), int(iu[1][o])
            if a not in alive or b not in alive:
                continue
            ok, rows = try_merge(models[a], models[b])
            if not ok:
                continue
            na = {"qs": models[a]["qs"] + models[b]["qs"],
                  "idx": np.concatenate([models[a]["idx"],
                                         models[b]["idx"]]),
                  "val": np.concatenate([models[a]["val"],
                                         models[b]["val"]]),
                  "_rows": rows}
            alive.discard(a)
            alive.discard(b)
            models[a] = None
            models[b] = None
            models.append(na)
            alive.add(len(models) - 1)
            merged += 1
        models = [m for i, m in enumerate(models) if i in alive]
        rec = {"iteration": it, "models": len(models),
               "merges": merged,
               "iteration_s": round(time.time() - t_i),
               "elapsed_s": round(time.time() - t0)}
        print(json.dumps(rec), flush=True)
        json.dump(rec, open(f"{CLOS}/iter_{it}.json", "w"), indent=1)
        np.savez_compressed(
            f"{CLOS}/deltas_{it}.npz",
            **{f"m{i:05d}_{sfx}": arr for i, m in enumerate(models)
               for sfx, arr in (("idx", m["idx"]),
                                ("val", m["val"]))})
        json.dump([{"qs": m["qs"]} for m in models],
                  open(f"{CLOS}/models_{it}.json", "w"))
        if merged == 0:
            break
    sizes = sorted((len(m["qs"]) for m in models), reverse=True)
    out = {"fixed_point": True, "models": len(models),
           "sizes_top20": sizes[:20],
           "covered": sum(sizes),
           "singletons": sum(1 for s in sizes if s == 1),
           "iterations": it}
    json.dump(out, open(f"{CLOS}/summary.json", "w"), indent=1)
    print(json.dumps(out), flush=True)
    print("PAIR-CLOSURE-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
