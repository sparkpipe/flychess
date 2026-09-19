"""MoF section splitter: break each DEGM chapter pool into position-
clustered sections (~20-30 positions each) for dedicated section flies —
the operator's finer layer-0 granularity. Clustering on standardized
flyfeat vectors (pure numpy k-means, same as the regime study); section
pools written as DEGM_Ch{i}_s{j}.jsonl in the pool schema; .pre.npz
regenerated per section via the precomputer conventions.
"""
import glob
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/home/spec/chess-lab")
import chess
import fly_curriculum as fc
import flyfeat_cb

POOL_DIR = "/home/spec/chess-lab/tbpools"
TARGET = 25          # positions per section (approx)
MIN_K = 2


def kmeans(X, K, iters=30, seed=0):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), K, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for k in range(K):
            m = lab == k
            if m.sum():
                C[k] = X[m].mean(0)
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def main():
    flyfeat_cb.feat_vec(chess.Board())
    for path in sorted(glob.glob(f"{POOL_DIR}/DEGM_Ch*.jsonl")):
        if "_s" in os.path.basename(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        ch = os.path.basename(path).replace(".jsonl", "")
        X = np.stack([flyfeat_cb.feat_vec(chess.Board(e["fen"]))[0]
                      for e in rows]).astype(np.float32)
        Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
        K = max(MIN_K, round(len(rows) / TARGET))
        K = min(K, len(rows))
        lab = kmeans(Xn, K, seed=hash(ch) % 1000)
        counts = np.bincount(lab, minlength=K)
        for k in range(K):
            if counts[k] < 5:
                continue
            out = f"{POOL_DIR}/{ch}_s{k}.jsonl"
            with open(out, "w") as f:
                for i, e in enumerate(rows):
                    if lab[i] == k:
                        e2 = dict(e, pool=f"{ch}_s{k}")
                        f.write(json.dumps(e2) + "\n")
        print(f"{ch}: {len(rows)} rows -> "
              f"{sum(1 for c in counts if c >= 5)} sections "
              f"(sizes {sorted(int(c) for c in counts if c >= 5)})",
              flush=True)
    print("MOF-SECTIONS-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
