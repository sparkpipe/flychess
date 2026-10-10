"""PHASE 3 — repair-merge the 28 instant lumps (operator directive
2026-09-21).

Rounds: pair lumps in descending delta-cosine; superpose; if the union
gates perfectly, merge free. Otherwise REPAIR: train only on the
failing questions plus a small passing sample (stage-6 maintenance
law — cost proportional to failures), re-gate the full union every 10
steps, cap 300 steps per pair. Merged models carry the ACTUAL
post-repair delta, re-sparsified top-4096. Iterate rounds to a fixed
point (a round with zero merges). Checkpoint per round: singles/p3/.

Runs on the GPU in the sweep's spare memory (GATE_CH=512 keeps the
gate activations inside the ~4GB headroom).
"""
import sys
import os
import json
import random
import time

os.environ.setdefault("ANORM", "1")
os.environ.setdefault("B", "1")
os.environ.setdefault("GATE_CH", "512")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True")
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc
from scipy import sparse

CLOS = "/home/spec/chess-lab/singles/closure"
P3 = "/home/spec/chess-lab/singles/p3"
CAP = int(os.environ.get("P3_CAP", "30"))   # scan = a few steps; deep forcing is a later pass
LR = 3e-4
GATE_EVERY = 10
SAMPLE_PASS = 10
K = 4096
os.makedirs(P3, exist_ok=True)


def load_models():
    z = np.load(f"{CLOS}/deltas_2.npz")
    state = json.load(open(f"{CLOS}/models_2.json"))
    models = []
    for i, m in enumerate(state):
        models.append({"qs": [tuple(q) for q in m["qs"]],
                       "idx": z[f"m{i:05d}_idx"],
                       "val": z[f"m{i:05d}_val"]})
    return models


def cos_matrix(models):
    n = len(models)
    indptr = np.zeros(n + 1, dtype=np.int64)
    for i, m in enumerate(models):
        indptr[i + 1] = indptr[i] + len(m["idx"])
    idx = np.concatenate([m["idx"] for m in models])
    val = np.concatenate([m["val"] for m in models])
    A = sparse.csr_matrix((val, idx, indptr),
                          shape=(n, int(idx.max()) + 1))
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    norms[norms < 1e-12] = 1.0
    A.data /= np.repeat(norms, np.diff(indptr))
    return (A @ A.T).toarray()


def main():
    flyfeat_cb.feat_vec(chess.Board())
    models = load_models()
    # P3_SKIP_BIG=1: exclude the largest lump from pairing (it is
    # carried through untouched — operator directive 2026-09-21:
    # merge the smaller clumps first, end state = big + second + ...)
    big = None
    if os.environ.get("P3_SKIP_BIG") == "1" and len(models) > 1:
        big = max(range(len(models)), key=lambda i: len(models[i]["qs"]))
        models[big]["skip"] = True
    print(json.dumps({"models_in": len(models),
                      "skip_big": models[big]["qs"][0] if big is not None
                      else None}), flush=True)
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

    def apply(deltas):
        with torch.no_grad():
            for k, p in base.named_parameters():
                p.copy_(base_state[k])
            flat = base_cat.clone()
            for m in deltas:
                flat.index_add_(0, torch.from_numpy(m["idx"]).to(fc.DEV),
                                torch.from_numpy(m["val"]).to(fc.DEV))
            off = 0
            for _, p in base.named_parameters():
                n_ = p.numel()
                p.copy_(flat[off:off + n_].view_as(p))
                off += n_

    def gate_rows(rows):
        pair, _, fails = fc.gate_tb(base, rows, random.Random(777),
                                    exhaustive=True)
        return pair, fails

    t0 = time.time()
    rnd = 0
    while True:
        rnd += 1
        t_r = time.time()
        S = cos_matrix(models)
        n = len(models)
        iu = np.triu_indices(n, 1)
        # ascending union size (small pairs first — min work, quick
        # wins on CPU); cosine desc as tiebreak
        union = np.array([len(models[a_]["qs"]) + len(models[b_]["qs"])
                          for a_, b_ in zip(iu[0], iu[1])])
        cosv = S[iu]
        order = np.lexsort((-cosv, union))
        alive = set(range(n))
        merges = 0
        work = []
        for o in order:
            a, b = int(iu[0][o]), int(iu[1][o])
            if a not in alive or b not in alive:
                continue
            if models[a].get("skip") or models[b].get("skip"):
                continue
            ma, mb = models[a], models[b]
            qs = ma["qs"] + mb["qs"]
            print(json.dumps({"pair": [len(ma["qs"]), len(mb["qs"])],
                              "cos": round(float(S[a, b]), 3)}),
                  flush=True)
            rows = [row_for(pl, fn) for pl, fn in qs]
            apply([ma, mb])
            pair, fails = gate_rows(rows)
            steps = 0
            ok = pair >= 0.999
            if not ok:
                fail_fens = set(fails) if isinstance(fails, dict) \
                    else set()
                rng = random.Random(rnd * 13 + 1)
                opt = torch.optim.Adam(base.parameters(), lr=LR)
                while steps < CAP:
                    rep_rows = [r for r in rows
                                if r["fen"] in fail_fens]
                    pass_rows = [r for r in rows
                                 if r["fen"] not in fail_fens]
                    rep_rows += rng.sample(
                        pass_rows, min(SAMPLE_PASS, len(pass_rows)))
                    for r in rep_rows:
                        fc.tb_step(base, opt, [r], rng)
                    steps += GATE_EVERY
                    pair, fails = gate_rows(rows)
                    if isinstance(fails, dict):
                        fail_fens = set(fails)
                    if pair >= 0.999:
                        ok = True
                        break
            if not ok and steps >= CAP:
                print(json.dumps({"stuck": [len(ma["qs"]),
                                            len(mb["qs"])],
                                  "steps": steps}), flush=True)
            if ok:
                merges += 1
                work.append(steps)
                alive.discard(a)
                alive.discard(b)
                with torch.no_grad():
                    dcat = torch.cat(
                        [p.detach().flatten() for _, p in
                         base.named_parameters()]) - base_cat
                v, idx = torch.topk(dcat.abs(), K)
                models.append({"qs": qs,
                               "idx": idx.cpu().numpy().astype(np.int64),
                               "val": dcat[idx].cpu().numpy()
                               .astype(np.float32)})
                alive.add(len(models) - 1)
                print(json.dumps({"merge": [len(ma["qs"]),
                                            len(mb["qs"])],
                                  "union": len(qs), "steps": steps,
                                  "cos": round(float(S[a, b]), 3)}),
                      flush=True)
        models = [m for i, m in enumerate(models) if i in alive]
        if any(m.get("skip") for m in models):
            models = [m for m in models if not m.get("skip")] \
                + [m for m in models if m.get("skip")]
        rec = {"round": rnd, "models": len(models), "merges": merges,
               "repair_steps": work,
               "round_s": round(time.time() - t_r),
               "elapsed_s": round(time.time() - t0)}
        print(json.dumps(rec), flush=True)
        json.dump(rec, open(f"{P3}/round_{rnd}.json", "w"), indent=1)
        np.savez_compressed(
            f"{P3}/deltas_{rnd}.npz",
            **{f"m{i:05d}_{sfx}": arr for i, m in enumerate(models)
               for sfx, arr in (("idx", m["idx"]),
                                ("val", m["val"]))})
        json.dump([{"qs": m["qs"]} for m in models],
                  open(f"{P3}/models_{rnd}.json", "w"))
        if merges == 0:
            break
    sizes = sorted((len(m["qs"]) for m in models), reverse=True)
    out = {"fixed_point": True, "models": len(models),
           "sizes": sizes, "covered": sum(sizes)}
    json.dump(out, open(f"{P3}/summary.json", "w"), indent=1)
    print(json.dumps(out), flush=True)
    print("P3-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
