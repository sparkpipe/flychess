"""Build the trainable H01 substrate from the extracted edge table.

From edgesV5 (63M directed edges with E/I type): subsample the top-K
postsynaptic-degree nodes, induce the signed subgraph, and emit a torch
sparse-CSR-ready npz: N nodes, edges (u, v, sign). sign = +1 excitatory
(typ 2), -1 inhibitory (typ 1).
"""
import glob
import json
import os
import sys
import numpy as np

PARTS = sorted(glob.glob(
    "/home/spec/chess-lab/h01_edges/edgesV5_part*.npz"))
OUT = "/home/spec/chess-lab/h01_graph.npz"
TOPK = int(os.environ.get("H01_TOPK", "200000"))


def main():
    pre = np.concatenate([np.load(f)["pre"] for f in PARTS])
    post = np.concatenate([np.load(f)["post"] for f in PARTS])
    typ = np.concatenate([np.load(f)["typ"] for f in PARTS])
    print(json.dumps({"edges": len(pre)}), flush=True)

    # node universe: union of both sides, compact ids
    nodes = np.unique(np.concatenate([pre, post]))
    nid = {int(n): i for i, n in enumerate(nodes.tolist())}
    u = np.fromiter((nid[int(x)] for x in pre), dtype=np.int32,
                    count=len(pre))
    v = np.fromiter((nid[int(x)] for x in post), dtype=np.int32,
                    count=len(post))
    del nid
    sign = np.where(typ == 2, 1.0, -1.0).astype(np.float32)

    # top-K by IN-degree within the universe
    indeg = np.bincount(v, minlength=len(nodes))
    top = np.argsort(-indeg)[:TOPK]
    keep = np.zeros(len(nodes), dtype=bool)
    keep[top] = True
    m = keep[u] & keep[v]
    uu, vv, ss = u[m], v[m], sign[m]
    # compact to 0..K-1
    remap = np.full(len(nodes), -1, dtype=np.int32)
    remap[top] = np.arange(len(top), dtype=np.int32)
    uu, vv = remap[uu], remap[vv]
    deg_in = np.bincount(vv, minlength=len(top))
    deg_out = np.bincount(uu, minlength=len(top))
    print(json.dumps({"nodes": len(top), "induced_edges": int(m.sum()),
                      "max_in": int(deg_in.max()),
                      "median_in": float(np.median(deg_in)),
                      "exc_frac": float((ss > 0).mean())}), flush=True)
    np.savez(OUT, u=uu, v=vv, sign=ss, n=len(top))
    print("H01-GRAPH-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
