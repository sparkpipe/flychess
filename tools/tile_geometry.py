"""TILE GEOMETRY (CPU only — runs beside the GPU bank).

Inputs: witness/pos_x.npy (corpus features), closure/models_2.json
(family membership), lichess_endgame_eval.jsonl (off-book probe set).

Produces the tiling map in feature space:
  - per-family centroid, radius (mean/max cosine distance of members)
  - nearest-neighbor tile distances; halfway-gap prediction
    (tile too small to reach midpoint: 2*radius < NN distance)
  - corpus coverage prediction: every position's nearest centroid
    within that family's radius -> % covered, gap list
  - PCA spectrum (effective dimensionality of endgame space)
  - Lichess off-book distances vs corpus distribution (why the bank
    flatlined, geometrically)
Output: singles/stretch/tile_geometry.json
"""
import sys
import os
import json

sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import numpy as np
import chess
import flyfeat_cb

WIT = "/home/spec/chess-lab/singles/witness"
CLOS = "/home/spec/chess-lab/singles/closure"
OUT = "/home/spec/chess-lab/singles/stretch/tile_geometry.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def main():
    flyfeat_cb.feat_vec(chess.Board())
    X = np.load(f"{WIT}/pos_x.npy").astype(np.float32)
    meta = json.load(open(f"{WIT}/meta.json"))
    fens = meta["fens"]
    pools = meta["pool"]
    row_of = {f: i for i, f in enumerate(fens)}
    state = json.load(open(f"{CLOS}/models_2.json"))

    fams = []
    for m in state:
        idx = [row_of[f] for _, f in m["qs"] if f in row_of]
        if idx:
            fams.append(idx)
    print(json.dumps({"families": len(fams),
                      "positions": len(fens)}))

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cents = np.stack([Xn[idx].mean(0) for idx in fams])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
    S = Xn @ cents.T                        # (positions, families)

    # radii: cosine distance of members to own centroid
    radii_mean, radii_max = [], []
    for fi, idx in enumerate(fams):
        d = 1 - S[idx, fi]
        radii_mean.append(float(d.mean()))
        radii_max.append(float(d.max()))
    radii_mean = np.array(radii_mean)
    radii_max = np.array(radii_max)

    # inter-centroid distances
    C = cents @ cents.T
    D = 1 - C
    np.fill_diagonal(D, 2.0)
    nn = np.argsort(D, axis=1)[:, :3]

    # halfway-gap prediction
    gaps = []
    for fi in range(len(fams)):
        d_nn = D[fi, nn[fi, 0]]
        if 2 * radii_mean[fi] < d_nn:
            gaps.append({"family": fi,
                         "radius_mean": round(float(radii_mean[fi]), 4),
                         "nn_dist": round(float(d_nn), 4),
                         "nn_family": int(nn[fi, 0])})

    # coverage prediction: nearest centroid within its radius
    nearest = S.argmax(1)
    cov_mean = cov_max = 0
    uncovered = []
    for pi in range(len(fens)):
        fi = nearest[pi]
        d = 1 - S[pi, fi]
        if d <= radii_mean[fi]:
            cov_mean += 1
        elif d <= radii_max[fi]:
            cov_max += 1
        else:
            uncovered.append({"pos": pi, "family": int(fi),
                              "d": round(float(d), 4),
                              "r_max": round(float(radii_max[fi]),
                                             4)})
    # PCA spectrum
    Xc = Xn - Xn.mean(0)
    ev = np.linalg.svd(Xc, compute_uv=False)[:25] ** 2
    spec = (ev / ev.sum()).tolist()

    # Lichess off-book distances
    lz = [json.loads(l) for l in open(
        "/home/spec/chess-lab/tbpools/lichess_endgame_eval.jsonl")]
    L = np.stack([flyfeat_cb.feat_vec(chess.Board(p["fen"]))[0]
                  for p in lz]).astype(np.float32)
    L /= np.linalg.norm(L, axis=1, keepdims=True) + 1e-9
    SL = L @ cents.T
    l_near = SL.max(1)
    c_near = S.max(1)          # corpus nearest-centroid cosine
    pct = lambda a, q: round(float(np.percentile(a, q)), 4)

    out = {
        "families": len(fams),
        "radius_mean": {
            "mean_of_means": round(float(radii_mean.mean()), 4),
            "median": round(float(np.median(radii_mean)), 4),
            "p90": round(float(np.percentile(radii_mean, 90)), 4)},
        "halfway_gap_families": len(gaps),
        "gap_examples": gaps[:10],
        "coverage_pred_mean_radius": round(cov_mean / len(fens), 4),
        "coverage_pred_max_radius": round(
            (cov_mean + cov_max) / len(fens), 4),
        "uncovered_n": len(uncovered),
        "uncovered_examples": uncovered[:10],
        "pca_spectrum_top10": [round(s, 4) for s in spec[:10]],
        "pca_dims_for_90pct": int(np.searchsorted(
            np.cumsum(spec), 0.90) + 1),
        "corpus_nn_cosine": {"median": pct(c_near, 50),
                             "p10": pct(c_near, 10)},
        "lichess_nn_cosine": {"median": pct(l_near, 50),
                              "p10": pct(l_near, 10),
                              "p90": pct(l_near, 90)},
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
