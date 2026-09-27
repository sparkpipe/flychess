"""Weight-space grouping over the singles-sweep solution vectors —
CPU only, runs beside the GPU sweep.

Inputs: singles/records.jsonl (solved rows) + singles/vec_{gi}.npz
(signed top-4096 delta, index space = concatenated named params) and
singles_repair/records.jsonl (repair vectors supersede their rows).

Outputs (singles/groups.json + stdout):
  - pairwise support-overlap and edit-cosine distributions
  - threshold-graph weight groups (components at cos >= T)
  - group composition by pool/cat; most-similar / most-orthogonal pairs
"""
import sys
import os
import json

sys.path.insert(0, "/home/spec/chess-lab")
import numpy as np
from scipy import sparse

IN = "/home/spec/chess-lab/singles"
REP = "/home/spec/chess-lab/singles_repair"
DIM = 74_060_515          # W_sens.weight 26933*2746 + biases; upper bound


def main():
    rep = {}
    if os.path.exists(f"{REP}/records.jsonl"):
        for line in open(f"{REP}/records.jsonl"):
            r = json.loads(line)
            rep[r["fen"]] = r
    rows = []
    for line in open(f"{IN}/records.jsonl"):
        r = json.loads(line)
        if r["stable_step"] is None and r["fen"] in rep:
            continue                     # superseded by repair record
        rows.append(r)
    print(f"vectors to group: {len(rows)}", flush=True)

    indptr = [0]
    indices = []
    data = []
    meta = []
    for r in rows:
        src = REP if (r["stable_step"] is None
                      and r["fen"] in rep) else IN
        gi = rep[r["fen"]]["gi"] if src == REP else r["gi"]
        z = np.load(f"{src}/vec_{gi}.npz")
        indices.append(z["idx"].astype(np.int64))
        data.append(z["val"].astype(np.float32))
        indptr.append(indptr[-1] + len(z["idx"]))
        meta.append({"gi": r["gi"], "fen": r["fen"],
                     "pool": r["pool"], "cat": r["cat"],
                     "authority": r.get("authority"),
                     "ply": r.get("ply")})
    idx = np.concatenate(indices)
    val = np.concatenate(data)
    indptr = np.array(indptr, dtype=np.int64)
    N = len(rows)
    D = int(idx.max()) + 1
    print(f"dim={D} nnz={len(idx)}", flush=True)

    A = sparse.csr_matrix((val, idx, indptr), shape=(N, D))
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    norms[norms < 1e-12] = 1.0
    A.data /= np.repeat(norms, np.diff(indptr))
    A = A.tocsr()
    S = (A @ A.T).toarray()
    B = sparse.csr_matrix((np.ones(len(idx), np.float32), idx, indptr),
                          shape=(N, D))
    O = (B @ B.T).toarray()              # |support intersection|
    K = (indptr[1:] - indptr[:-1]).astype(np.float32)
    Ov = O / np.minimum(K[:, None], K[None, :])

    iu = np.triu_indices(N, 1)
    cos = S[iu]
    ov = Ov[iu]
    prof = {
        "n": N,
        "cos_mean": round(float(cos.mean()), 4),
        "cos_std": round(float(cos.std()), 4),
        "cos_p10": round(float(np.percentile(cos, 10)), 4),
        "cos_p50": round(float(np.percentile(cos, 50)), 4),
        "cos_p90": round(float(np.percentile(cos, 90)), 4),
        "ov_min_overlap@2048": round(float(ov.min()), 4),
        "ov_mean": round(float(ov.mean()), 4),
        "pairs_cos>0.5": int((cos > 0.5).sum()),
        "pairs_cos>0.3": int((cos > 0.3).sum()),
        "pairs_cos<-0.3": int((cos < -0.3).sum()),
    }
    # threshold components at several cos levels
    import scipy.sparse.csgraph as cg
    groups = {}
    for T in (0.2, 0.35, 0.5):
        adj = sparse.csr_matrix((S > T).astype(np.int8))
        ncomp, lab = cg.connected_components(adj, directed=False)
        sizes = np.bincount(lab)
        groups[f"cos{T}"] = {
            "components": int(ncomp),
            "largest": int(sizes.max()),
            "top5": sorted(sizes.tolist(), reverse=True)[:5],
            "singletonish<=2": int((sizes <= 2).sum())}
        if T == 0.35:
            np.save(f"{IN}/group_labels.npy", lab)

    # per-vector top-5 most similar
    top = np.argsort(-S + np.eye(N) * 10, axis=1)[:, :5]
    out = {"profile": prof, "groups": groups,
           "top_pairs": []}
    ii, jj = iu
    order = np.argsort(-cos)[:200]
    shown = 0
    for o in order:
        if shown >= 10:
            break
        if meta[ii[o]]["fen"] == meta[jj[o]]["fen"]:
            continue                     # duplicate FEN = same question
        out["top_pairs"].append(
            {"a": meta[ii[o]]["gi"], "b": meta[jj[o]]["gi"],
             "cos": round(float(cos[o]), 3),
             "ov": round(float(ov[o]), 3),
             "pools": [meta[ii[o]]["pool"], meta[jj[o]]["pool"]]})
        shown += 1
    np.save(f"{IN}/sim_matrix.npy", S.astype(np.float32))
    json.dump(meta, open(f"{IN}/vec_meta.json", "w"))
    json.dump(out, open(f"{IN}/groups.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print("GROUPING-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
