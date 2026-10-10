"""Position-level structure of the DEGM corpus, independent of SF
grades (runs while the grader works): transpositions/convergence and
feature-space clustering.

Answers the operator's framing: 'endings of many questions probably
combine, even if the start positions are different.'
"""
import os
import re
import sys
import json
import chess
import chess.pgn
from collections import Counter, defaultdict

sys.path.insert(0, "/home/spec/chess-lab")
import numpy as np
import flyfeat_cb

PGN = os.path.expanduser("~/chess-lab/books/DEGM.pgn")
OUT = "/home/spec/chess-lab/positions_profile.json"


def sane_board(b):
    return (b.is_valid() and not b.is_game_over()
            and b.king(chess.WHITE) is not None
            and b.king(chess.BLACK) is not None
            and list(b.legal_moves))


def kmeans_cos(X, k, iters=25, seed=0):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), k, replace=False)].astype(np.float32)
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
    pos = defaultdict(list)
    with open(PGN, encoding="utf-8", errors="replace") as f:
        gi = 0
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            gi += 1
            fen = g.headers.get("FEN")
            if not fen:
                continue
            try:
                b = chess.Board(fen)
            except Exception:
                continue
            if not sane_board(b):
                continue
            pos[b.fen()].append((gi, 0))
            ply = 0
            for n in g.mainline():
                try:
                    b.push(n.move)
                except Exception:
                    break
                ply += 1
                if sane_board(b):
                    pos[b.fen()].append((gi, ply))

    total = sum(len(v) for v in pos.values())
    distinct = len(pos)
    multi = {f: v for f, v in pos.items() if len(v) > 1}
    cross = {f: v for f, v in multi.items()
             if len({e for e, _ in v}) > 1}
    ply_of = {f: max(p for _, p in v) for f, v in pos.items()}
    deep_cross = [f for f in cross if ply_of[f] >= 6]

    fens = list(pos)
    X = np.stack([flyfeat_cb.feat_vec(chess.Board(f))[0]
                  for f in fens]).astype(np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    lab = kmeans_cos(Xn, 40)
    sizes = Counter(lab.tolist())
    # convergence per cluster: how many of a cluster's positions are
    # reached from >1 example (the 'endings combine' measure)
    cluster_cross = {}
    for j in range(40):
        members = [fens[i] for i in np.where(lab == j)[0]]
        nc = sum(1 for f in members if f in cross)
        cluster_cross[j] = {"size": len(members),
                            "cross_example": nc}

    prof = {"positions_total": total, "distinct": distinct,
            "reached_from_multiple_plies": len(multi),
            "cross_example_positions": len(cross),
            "deep_cross_ply6plus": len(deep_cross),
            "examples": len({e for v in pos.values() for e, _ in v}),
            "cluster_sizes_top10":
                sizes.most_common(10),
            "cluster_cross_top10": sorted(
                cluster_cross.items(),
                key=lambda kv: -kv[1]["cross_example"])[:10]}
    json.dump(prof, open(OUT, "w"), indent=1)
    print(json.dumps(prof, indent=1))


if __name__ == "__main__":
    main()
