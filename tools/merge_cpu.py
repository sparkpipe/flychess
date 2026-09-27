"""Merge tests on real corpus vectors — CPU ONLY (runs beside the GPU
sweep; CUDA is hidden and FCDEV=cpu).

Samples from the saved similarity matrix:
  5 pairs  cos >  0.6   (aligned edits)
  5 pairs |cos| < 0.15  (orthogonal edits)
  5 pairs  cos < -0.35  (opposing edits)
  1 four-way and 1 six-way merge from the highest-average-cosine set
For each: superpose (init + sum of deltas) -> exhaustive gate every
constituent question -> interleaved repair (cap 300 steps) -> steps to
all-pass. Duplicate-FEN pairs are skipped (same question twice).
"""
import sys
import os
import json
import random

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

torch.set_num_threads(16)
IN = "/home/spec/chess-lab/singles"
REP = "/home/spec/chess-lab/singles_repair"
OUT = "/home/spec/chess-lab/merge_cpu.json"
CAP = 300
LR = 3e-4
GATE_EVERY = 10
STABLE_STREAK = 5


def load_vecs(gis):
    S = np.load(f"{IN}/sim_matrix.npy")
    meta = json.load(open(f"{IN}/vec_meta.json"))
    vecs = {}
    for gi in gis:
        src = IN
        z = f"{IN}/vec_{gi}.npz"
        vecs[gi] = (np.load(z), src)
    return S, meta, vecs


def row_by_fen(pool, fen, cache):
    if pool not in cache:
        cache[pool] = {r["fen"]: r for r in fc.load_pools([pool])}
    return cache[pool][fen]


def main():
    flyfeat_cb.feat_vec(chess.Board())
    S = np.load(f"{IN}/sim_matrix.npy")
    meta = json.load(open(f"{IN}/vec_meta.json"))
    fens = [m["fen"] for m in meta]
    N = len(meta)
    iu = np.triu_indices(N, 1)
    cos = S[iu]
    fena = np.array([fens[i] for i in iu[0]])
    fenb = np.array([fens[j] for i, j in zip(iu[0], iu[1])])
    dup = fena == fenb
    rng = random.Random(7)

    def sample(lo, hi, k):
        sel = np.where((cos > lo) & (cos < hi) & ~dup)[0]
        pick = rng.sample(sorted(sel.tolist()), k)
        return [(int(iu[0][p]), int(iu[1][p]), float(cos[p]))
                for p in pick]

    def pair(lo, hi, k):
        return [(i, j, c) for i, j, c in sample(lo, hi, k)]
    tests = ([("aligned", [p[0], p[1]], p[2], None)
              for p in pair(0.6, 1.01, 5)]
             + [("orthogonal", [p[0], p[1]], p[2], None)
                for p in pair(-0.15, 0.15, 5)]
             + [("opposing", [p[0], p[1]], p[2], None)
                for p in pair(-1.01, -0.35, 5)])

    # multi-way: highest-average-cosine distinct-fen members
    avg = (S.sum(1) - 1) / (N - 1)
    order = np.argsort(-avg)
    seen, clique = set(), []
    for i in order:
        if fens[i] in seen:
            continue
        seen.add(fens[i])
        clique.append(int(i))
        if len(clique) == 6:
            break
    tests.append(("four-way", clique[:4], None, None))
    tests.append(("six-way", clique, None, None))

    cache = {}
    rep_meta = {}
    if os.path.exists(f"{REP}/records.jsonl"):
        for line in open(f"{REP}/records.jsonl"):
            r = json.loads(line)
            rep_meta[r["fen"]] = r["gi"]
    results = []
    for tag, gis, c, _ in tests:
        rows = []
        for gi in gis:
            fen = fens[gi]
            pool = meta[gi]["pool"]
            rows.append(row_by_fen(pool, fen, cache))
        m = tp.build_model(0)
        with torch.no_grad():
            for gi in gis:
                z = np.load(f"{IN}/vec_{gi}.npz")
                idx = torch.from_numpy(z["idx"].astype(np.int64))
                val = torch.from_numpy(z["val"].astype(np.float32))
                flat = torch.cat(
                    [p.detach().flatten()
                     for _, p in m.named_parameters()])
                flat.index_add_(0, idx, val)
                off = 0
                for _, p in m.named_parameters():
                    n = p.numel()
                    p.add_(flat[off:off + n].view_as(p))
                    off += n
        def allgates():
            return {f"gi{gi}": round(
                fc.gate_tb(m, [rows[k]], random.Random(777),
                           exhaustive=True)[0], 3)
                for k, gi in enumerate(gis)}
        g0 = allgates()
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        rngs = random.Random(1)
        steps = 0
        traj = [g0]
        while steps < CAP:
            if all(v >= 0.98 for v in g0.values()):
                break
            for k in range(len(gis)):
                fc.tb_step(m, opt, [rows[k]], rngs)
            steps += 1
            if steps % GATE_EVERY == 0:
                g0 = allgates()
                traj.append(g0)
                if all(v >= 0.98 for v in g0.values()):
                    break
        passed = all(v >= 0.98 for v in g0.values())
        rec = {"test": tag, "gis": gis,
               "cos": round(c, 3) if c is not None else None,
               "superposed_pass": all(v >= 0.98
                                      for v in traj[0].values()),
               "steps_to_allpass": steps if passed else None,
               "final": g0}
        results.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print("MERGE-CPU-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
