"""TILING PoC (quick, operator request): does skill radius follow
geometric radius?

Uses the ~40 BUILT distbank flies as tiles; probes are positions from
UNBUILT families (never trained on by these tiles). For each probe:
nearest built tile by feature centroid, distance vs that tile's radii
(mean/max from its OWN members), then gate the tile's fly on the
probe. Verdict bands:
  inside  (d <= r_mean)   -> should be mostly correct
  border  (r_mean < d <= r_max)
  outside (d > r_max)     -> should be mostly wrong
Monotone accuracy across bands = tiling generalizes, and the radius
law is quantitative.
"""
import sys
import os
import json
import random

os.environ.setdefault("ANORM", "1")
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

DB = "/home/spec/chess-lab/singles/distbank"
CLOS = "/home/spec/chess-lab/singles/closure"
WIT = "/home/spec/chess-lab/singles/witness"
PER_BAND = int(os.environ.get("PER_BAND", "60"))


def main():
    flyfeat_cb.feat_vec(chess.Board())
    built = {}
    for line in open(f"{DB}/records.jsonl"):
        r = json.loads(line)
        if r.get("status") == "ok":
            built[r["lump"]] = r
    print(json.dumps({"built_tiles": len(built)}), flush=True)

    meta = json.load(open(f"{WIT}/meta.json"))
    fens, pools = meta["fens"], meta["pool"]
    row_of = {f: i for i, f in enumerate(fens)}
    state = json.load(open(f"{CLOS}/models_2.json"))
    fam_of_pos = {}
    built_members = set()
    for li, m in enumerate(state):
        for pl, f in m["qs"]:
            if f in row_of:
                fam_of_pos[row_of[f]] = li
                if li in built:
                    built_members.add(row_of[f])

    X = np.load(f"{WIT}/pos_x.npy").astype(np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cents, r_mean, r_max = {}, {}, {}
    for li in built:
        idx = [row_of[f] for _, f in state[li]["qs"] if f in row_of]
        c = Xn[idx].mean(0)
        c /= np.linalg.norm(c) + 1e-9
        cents[li] = c
        d = 1 - Xn[idx] @ c
        r_mean[li] = float(d.mean())
        r_max[li] = float(d.max())
    C = np.stack([cents[li] for li in built])
    lis = list(built)

    # probes: positions of UNBUILT families only
    probes = [i for i in range(len(fens))
              if fam_of_pos.get(i) not in built]
    print(json.dumps({"probe_pool": len(probes)}), flush=True)
    rng = random.Random(3)
    bands = {"inside": [], "border": [], "outside": []}
    rng.shuffle(probes)
    for pi in probes:
        s = Xn[pi] @ C.T
        k = int(np.argmax(s))
        li = lis[k]
        d = 1 - float(s[k])
        b = "inside" if d <= r_mean[li] else (
            "border" if d <= r_max[li] else "outside")
        if len(bands[b]) < PER_BAND:
            bands[b].append((pi, li, d))

    pool_cache = {}

    def row_for(pool, fen):
        if pool not in pool_cache:
            pool_cache[pool] = {r["fen"]: r
                                for r in fc.load_pools([pool])}
        return pool_cache[pool][fen]

    base = tp.build_model(0)
    base_sd = {k: v.detach().cpu().clone()
               for k, v in base.named_parameters()}
    base_cat = torch.cat([base_sd[k].flatten()
                          for k, _ in base.named_parameters()])
    vec_cache = {}

    def apply_fly(li):
        if li not in vec_cache:
            z = np.load(f"{DB}/vec_{li}.npz")
            vec_cache[li] = (z["idx"], z["val"])
        idx, val = vec_cache[li]
        with torch.no_grad():
            for k, p in base.named_parameters():
                p.copy_(base_sd[k])
            flat = base_cat.to(fc.DEV)
            flat.index_add_(0, torch.from_numpy(idx).to(fc.DEV),
                            torch.from_numpy(val).to(fc.DEV))
            off = 0
            for _, p in base.named_parameters():
                n_ = p.numel()
                p.copy_(flat[off:off + n_].view_as(p))
                off += n_

    results = {b: {"n": 0, "ok": 0, "d": []} for b in bands}
    for b, items in bands.items():
        for pi, li, d in items:
            row = row_for(pools[pi], fens[pi])
            apply_fly(li)
            pair, _, _ = fc.gate_tb(base, [row], random.Random(777),
                                    exhaustive=True)
            results[b]["n"] += 1
            results[b]["ok"] += int(pair >= 0.98)
            results[b]["d"].append(round(d, 4))
    out = {b: {"n": v["n"],
               "acc": round(v["ok"] / max(v["n"], 1), 3),
               "d_median": round(float(np.median(v["d"])), 4)
               if v["d"] else None}
           for b, v in results.items()}
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open("/home/spec/chess-lab/singles/stretch/"
                        "tile_poc.json", "w"), indent=1)
    print("TILE-POC-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
