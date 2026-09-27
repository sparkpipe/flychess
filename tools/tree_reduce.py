"""TREE REDUCTION (operator directive 2026-09-20): pairwise merges with
minimal expected work, winners re-paired each round, coverage doubling
per round — measures how far merge size can grow before repair cost
blows up.

Round protocol:
  - pair models greedily by delta-cosine (nearest neighbor = min
    expected work pairing)
  - merged init = seed-0 init + sum of the two sparse deltas
  - superpose -> exhaustive gate every constituent question
  - if not all-pass: interleaved repair over the union set, gate every
    10, cap 300 steps ("min work": stop at all-pass)
  - carried model = ACTUAL post-repair delta, re-sparsified top-4096
  - checkpoint every round: singles/tree/ (stats + all deltas)

CPU only: FCDEV=cpu, CUDA hidden, nice. Env TREE_LEAVES=N runs a
random-leaf pilot; default = all unique-fen solved vectors.
"""
import sys
import os
import json
import random
import time

os.environ["FCDEV"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
import scipy.sparse.csgraph as cg

torch.set_num_threads(16)
IN = "/home/spec/chess-lab/singles"
REP = "/home/spec/chess-lab/singles_repair"
TREE = "/home/spec/chess-lab/singles/tree"
LEAVES = int(os.environ.get("TREE_LEAVES", "0"))
CAP = int(os.environ.get("TREE_CAP", "300"))
LR = 3e-4
GATE_EVERY = 10
K = 4096


def load_leaves():
    rep = {}
    if os.path.exists(f"{REP}/records.jsonl"):
        for line in open(f"{REP}/records.jsonl"):
            r = json.loads(line)
            rep[r["fen"]] = r
    seen = set()
    leaves = []
    for line in open(f"{IN}/records.jsonl"):
        r = json.loads(line)
        fen = r["fen"]
        if fen in seen:
            continue
        solved = r["stable_step"] is not None
        if fen in rep:
            pass                          # repair vector wins
        elif not solved:
            continue
        seen.add(fen)
        src = REP if fen in rep else IN
        gi = rep[fen]["gi"] if fen in rep else r["gi"]
        z = np.load(f"{src}/vec_{gi}.npz")
        leaves.append({"pool": r["pool"], "fen": fen,
                       "idx": z["idx"].astype(np.int64),
                       "val": z["val"].astype(np.float32)})
    return leaves


def cos_matrix(models):
    n = len(models)
    indptr = np.zeros(n + 1, dtype=np.int64)
    for i, m in enumerate(models):
        indptr[i + 1] = indptr[i] + len(m["delta"]["idx"])
    idx = np.concatenate([m["delta"]["idx"] for m in models])
    val = np.concatenate([m["delta"]["val"] for m in models])
    A = sparse.csr_matrix((val, idx, indptr),
                          shape=(n, int(idx.max()) + 1))
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    norms[norms < 1e-12] = 1.0
    A.data /= np.repeat(norms, np.diff(indptr))
    return (A @ A.T).toarray()


def greedy_pairs(S):
    """greedy max-cosine matching over the similarity matrix."""
    n = S.shape[0]
    iu = np.triu_indices(n, 1)
    order = np.argsort(-S[iu])
    used = np.zeros(n, bool)
    pairs = []
    for o in order:
        a, b = int(iu[0][o]), int(iu[1][o])
        if used[a] or used[b]:
            continue
        used[a] = used[b] = True
        pairs.append((a, b, float(S[a, b])))
    for i in range(n):
        if not used[i]:
            pairs.append((i, -1, 0.0))
    return pairs


def main():
    flyfeat_cb.feat_vec(chess.Board())
    os.makedirs(TREE, exist_ok=True)
    leaves = load_leaves()
    if LEAVES:
        rnd = random.Random(11)
        leaves = rnd.sample(leaves, min(LEAVES, len(leaves)))
    print(json.dumps({"leaves": len(leaves)}), flush=True)

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

    models = [{"qs": [(l["pool"], l["fen"])],
               "delta": {"idx": l["idx"], "val": l["val"]}}
              for l in leaves]
    random.Random(3).shuffle(models)
    round_k = 0
    t0 = time.time()
    while len(models) > 1:
        round_k += 1
        t_r = time.time()
        S = cos_matrix(models)
        pairs = greedy_pairs(S)
        newmodels = []
        stats = {"round": round_k, "n_in": len(models),
                 "n_merges": 0, "superposed": 0, "repaired": 0,
                 "fail": 0, "solo": 0, "steps": [],
                 "pair_cos_mean": round(float(np.mean(
                     [c for _, b_, c in pairs if b_ >= 0])), 3)}
        for a, b, c in pairs:
            if b < 0:
                newmodels.append(models[a])
                stats["solo"] += 1
                continue
            ma, mb = models[a], models[b]
            qs = ma["qs"] + mb["qs"]
            rows = [row_for(pl, fn) for pl, fn in qs]
            with torch.no_grad():
                for k, p in base.named_parameters():
                    p.copy_(base_state[k])
                flat = base_cat.clone()
                for m in (ma, mb):
                    flat.index_add_(0,
                                    torch.from_numpy(
                                        m["delta"]["idx"]),
                                    torch.from_numpy(
                                        m["delta"]["val"]))
                off = 0
                for _, p in base.named_parameters():
                    n_ = p.numel()
                    p.copy_(flat[off:off + n_].view_as(p))
                    off += n_
            def allgates():
                return all(fc.gate_tb(base, [r], random.Random(777),
                                      exhaustive=True)[0] >= 0.98
                           for r in rows)
            steps = 0
            ok = allgates()
            if ok:
                stats["superposed"] += 1
            else:
                opt = torch.optim.Adam(base.parameters(), lr=LR)
                rng = random.Random(round_k * 7 + 1)
                while steps < CAP:
                    for r in rows:
                        fc.tb_step(base, opt, [r], rng)
                    steps += 1
                    if steps % GATE_EVERY == 0 and allgates():
                        ok = True
                        break
            stats["n_merges"] += 1
            if ok:
                if steps:
                    stats["repaired"] += 1
                    stats["steps"].append(steps)
            else:
                stats["fail"] += 1
            with torch.no_grad():
                dcat = torch.cat([p.detach().flatten()
                                  for _, p in
                                  base.named_parameters()]) - base_cat
            v, idx = torch.topk(dcat.abs(), K)
            newmodels.append({
                "qs": qs,
                "delta": {"idx": idx.numpy().astype(np.int64),
                          "val": dcat[idx].numpy().astype(np.float32)}})
        sl = stats["steps"]
        stats["steps_mean"] = round(float(np.mean(sl)), 1) if sl else 0
        stats["steps_max"] = int(max(sl)) if sl else 0
        stats["steps"] = sl
        stats["cov"] = max(len(m["qs"]) for m in newmodels)
        stats["round_s"] = round(time.time() - t_r)
        stats["elapsed_s"] = round(time.time() - t0)
        models = newmodels
        json.dump(stats, open(f"{TREE}/round_{round_k}.json", "w"),
                  indent=1)
        np.savez_compressed(
            f"{TREE}/deltas_{round_k}.npz",
            **{f"m{i:05d}_{suffix}": arr
               for i, m in enumerate(models)
               for suffix, arr in (("idx", m["delta"]["idx"]),
                                   ("val", m["delta"]["val"]))})
        with open(f"{TREE}/models_{round_k}.json", "w") as f:
            json.dump([{"qs": m["qs"]} for m in models], f)
        print(json.dumps(stats), flush=True)
    print("TREE-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
