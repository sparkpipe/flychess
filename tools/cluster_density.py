"""Cluster-structure measurement over the distinct DEGM positions
(operator's hypothesis: ~20% large clusters, ~30% medium, ~50% small).

Neighborhood-density method (k-independent): for every distinct
position, count feature-space neighbors above cosine thresholds; bucket
positions by neighborhood size. Also reports kmeans partitions at
several k for the size-distribution view.
"""
import os
import sys
import json
import chess
import chess.pgn
from collections import defaultdict, Counter

sys.path.insert(0, "/home/spec/chess-lab")
import numpy as np
import flyfeat_cb

PGN = os.path.expanduser("~/chess-lab/books/DEGM.pgn")
OUT = "/home/spec/chess-lab/cluster_density.json"


def sane_board(b):
    return (b.is_valid() and not b.is_game_over()
            and b.king(chess.WHITE) is not None
            and b.king(chess.BLACK) is not None
            and list(b.legal_moves))


def kmeans_cos(X, k, iters=30, seed=0):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    lab = None
    for _ in range(iters):
        lab = (X @ C.T).argmax(1)
        for j in range(k):
            m = X[lab == j]
            if len(m):
                C[j] = m.mean(0)
                C[j] /= np.linalg.norm(C[j]) + 1e-9
    return lab


def main():
    flyfeat_cb.feat_vec(chess.Board())
    fens = set()
    with open(PGN, encoding="utf-8", errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            fen = g.headers.get("FEN")
            if not fen:
                continue
            try:
                b = chess.Board(fen)
            except Exception:
                continue
            if not sane_board(b):
                continue
            fens.add(b.fen())
            for n in g.mainline():
                try:
                    b.push(n.move)
                except Exception:
                    break
                if sane_board(b):
                    fens.add(b.fen())
    fens = sorted(fens)
    print(f"distinct positions: {len(fens)}", flush=True)
    X = np.stack([flyfeat_cb.feat_vec(chess.Board(f))[0]
                  for f in fens]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    n = len(X)

    # neighborhood density, chunked cosine
    dens = {t: np.zeros(n, dtype=np.int32)
            for t in (0.6, 0.7, 0.8, 0.9)}
    CH = 2048
    for s in range(0, n, CH):
        e = min(s + CH, n)
        S = X[s:e] @ X.T                    # (chunk, n)
        for t in dens:
            dens[t][s:e] = (S >= t).sum(1) - 1   # exclude self
    prof = {}
    for t, d in dens.items():
        large = int((d >= 200).sum())
        med = int(((d >= 20) & (d < 200)).sum())
        small = int((d < 20).sum())
        prof[f"cos{t}"] = {
            "large(>=200 nb)": round(100 * large / n, 1),
            "medium(20-199)": round(100 * med / n, 1),
            "small(<20)": round(100 * small / n, 1),
            "median_neighbors": int(np.median(d)),
            "p90_neighbors": int(np.percentile(d, 90))}

    # kmeans size distributions
    for k in (50, 150, 400):
        lab = kmeans_cos(X, k)
        sizes = np.bincount(lab, minlength=k)
        prof[f"kmeans{k}"] = {
            "top10_share_pct": round(100 * sizes[np.argsort(
                -sizes)][:10].sum() / n, 1),
            "clusters>=500": int((sizes >= 500).sum()),
            "clusters100-499": int(((sizes >= 100) &
                                    (sizes < 500)).sum()),
            "clusters<100": int((sizes < 100).sum()),
            "largest": int(sizes.max()),
            "singletons": int((sizes == 1).sum())}
    json.dump(prof, open(OUT, "w"), indent=1)
    print(json.dumps(prof, indent=1))


if __name__ == "__main__":
    main()
