"""Cluster-structure analysis of the Lichess puzzle DB (operator
directive), directly comparable to the DEGM profile.

Uniform random sample (reservoir) of the 6.1M-puzzle CSV; question
position = FEN after the opponent's setup move (Moves[0]); same
density buckets and kmeans resolutions as cluster_density.py; plus
per-cluster dominant THEMES and median rating (the Lichess advantage:
human-tagged ground truth for the taxonomy question).
"""
import os
import sys
import csv
import json
import random
import chess

sys.path.insert(0, "/home/spec/chess-lab")
import numpy as np
import flyfeat_cb

CSV = "/home/spec/chess-lab/puzzles/lichess_db_puzzle.csv"
OUT = "/home/spec/chess-lab/lichess_density.json"
N_SAMPLE = 40000


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
    rng = random.Random(0)
    res = []
    with open(CSV, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for i, row in enumerate(rd):
            if i % 7 != 0:                 # pre-thin: every 7th row
                continue
            if len(res) < N_SAMPLE:
                res.append(row)
            else:
                j = rng.randrange(i // 7 + 1)
                if j < N_SAMPLE:
                    res[j] = row
    print(f"sampled {len(res)} of 6.1M", flush=True)

    seen = {}
    n_bad = 0
    for row in res:
        try:
            b = chess.Board(row["FEN"])
            mvs = row["Moves"].split()
            if not mvs:
                n_bad += 1
                continue
            b.push(chess.Move.from_uci(mvs[0]))   # opponent setup
            if not b.is_valid() or b.king(chess.WHITE) is None \
                    or b.king(chess.BLACK) is None \
                    or b.is_game_over():
                n_bad += 1
                continue
            fen = b.fen()
            if fen not in seen:
                seen[fen] = row
        except Exception:
            n_bad += 1
    fens = sorted(seen)
    print(f"distinct question positions: {len(fens)} "
          f"(bad/skipped {n_bad}, dupes "
          f"{len(res) - n_bad - len(fens)})", flush=True)

    X = np.stack([flyfeat_cb.feat_vec(chess.Board(f))[0]
                  for f in fens]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    n = len(X)

    dens = {t: np.zeros(n, dtype=np.int32)
            for t in (0.6, 0.7, 0.8, 0.9)}
    CH = 2048
    for s in range(0, n, CH):
        e = min(s + CH, n)
        S = X[s:e] @ X.T
        for t in dens:
            dens[t][s:e] = (S >= t).sum(1) - 1
    prof = {"sample": len(res), "distinct": n}
    for t, d in dens.items():
        prof[f"cos{t}"] = {
            "large(>=200 nb)": round(100 * (d >= 200).sum() / n, 1),
            "medium(20-199)": round(
                100 * ((d >= 20) & (d < 200)).sum() / n, 1),
            "small(<20)": round(100 * (d < 20).sum() / n, 1),
            "median_neighbors": int(np.median(d)),
            "p90_neighbors": int(np.percentile(d, 90))}

    rows = [seen[f] for f in fens]
    ratings = np.array([float(r["Rating"]) for r in rows])
    META = {"opening", "middlegame", "endgame", "short", "long",
            "veryLong", "advantage", "crushing", "equality",
            "beginner", "intermediate", "advanced", "master",
            "masterLevel", "oneMove", "normal"}
    for k in (50, 150):
        lab = kmeans_cos(X, k)
        sizes = np.bincount(lab, minlength=k)
        top = np.argsort(-sizes)[:10]
        clusters = []
        for j in top:
            mem = np.where(lab == j)[0]
            tc = {}
            for i in mem:
                for th in rows[i]["Themes"].split():
                    if th not in META:
                        tc[th] = tc.get(th, 0) + 1
            dom = max(tc.items(), key=lambda kv: kv[1]) if tc \
                else ("?", 0)
            clusters.append({
                "size": int(sizes[j]),
                "dominant_theme": dom[0],
                "theme_share": round(dom[1] / len(mem), 2),
                "median_rating": int(np.median(ratings[mem])),
                "top_themes": sorted(tc.items(), key=lambda kv: -kv[1])[:4]})
        prof[f"kmeans{k}"] = {
            "top10_share_pct": round(
                100 * sizes[top].sum() / n, 1),
            "clusters>=500": int((sizes >= 500).sum()),
            "clusters100-499": int(((sizes >= 100) &
                                    (sizes < 500)).sum()),
            "clusters<100": int((sizes < 100).sum()),
            "largest": int(sizes.max()),
            "singletons": int((sizes == 1).sum()),
            "top_clusters": clusters}
    json.dump(prof, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in prof.items()
                      if not k.startswith("kmeans")}, indent=1),
          flush=True)
    print("LICHESS-DENSITY-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
