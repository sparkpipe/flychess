"""MoF ROUTING TEST (operator directive 2026-09-25).

N flies (variable via MOF_N; default = all closure flies). For a random
sample of questions, query EVERY fly on the position and record:
  - pick correctness (top move in the approved set)
  - score margin of the fly's top move (an INTERNAL signal)
  - feature-kNN cosine to the fly's member positions (EXTERNAL router)
  - answer agreement across flies (majority-vote signal)
Ranking functions are then evaluated: does the top-ranked fly answer
correctly (top-1), any of the top-5, mean rank of a correct fly.
Oracle ceiling = questions where >=1 fly knows (should be ~100% by
closure verification).

Usage: MOF_N=138 MOF_SAMPLE=100 python3 tools/mof_router.py
Output: singles/mof_test.json
"""
import sys
import os
import json
import random
import time

os.environ.setdefault("ANORM", "1")
os.environ.setdefault("B", "1")
os.environ.setdefault("GATE_CH", "512")
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

CLOS = "/home/spec/chess-lab/singles/closure"
MOF_N = int(os.environ.get("MOF_N", "0"))     # 0 = all
SAMPLE = int(os.environ.get("MOF_SAMPLE", "100"))
OUTP = "/home/spec/chess-lab/singles/mof_test.json"


def load_flies():
    z = np.load(f"{CLOS}/deltas_2.npz")
    state = json.load(open(f"{CLOS}/models_2.json"))
    flies = []
    for i, m in enumerate(state):
        flies.append({"qs": [tuple(q) for q in m["qs"]],
                      "idx": z[f"m{i:05d}_idx"],
                      "val": z[f"m{i:05d}_val"]})
    flies.sort(key=lambda f: -len(f["qs"]))
    if MOF_N:
        flies = flies[:MOF_N]
    return flies


def main():
    flyfeat_cb.feat_vec(chess.Board())
    flies = load_flies()
    print(json.dumps({"flies": len(flies)}), flush=True)

    # global feature bank over all member positions
    t0 = time.time()
    all_fens = []
    for f in flies:
        all_fens += [fen for _, fen in f["qs"]]
    fen_set = sorted(set(all_fens))
    fen_idx = {fen: i for i, fen in enumerate(fen_set)}
    F = np.stack([flyfeat_cb.feat_vec(chess.Board(f))[0]
                  for f in fen_set]).astype(np.float32)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    print(json.dumps({"feature_bank": len(fen_set),
                      "build_s": round(time.time() - t0)}), flush=True)
    for f in flies:
        f["midx"] = np.array([fen_idx[fen] for _, fen in f["qs"]])

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

    def apply_fly(f):
        with torch.no_grad():
            for k, p in base.named_parameters():
                p.copy_(base_state[k])
            flat = base_cat.clone()
            flat.index_add_(0, torch.from_numpy(f["idx"]).to(fc.DEV),
                            torch.from_numpy(f["val"]).to(fc.DEV))
            off = 0
            for _, p in base.named_parameters():
                n_ = p.numel()
                p.copy_(flat[off:off + n_].view_as(p))
                off += n_

    def query_fly(f, row):
        """fly's pick, margin, correctness on one row."""
        apply_fly(f)
        fvb, slots, tgts, cl, rich = fc.build_tb_batch(
            random.Random(1), [row], 1)
        sv, vv, vm = slots[0]
        L = len(sv)
        Fd = rich[0][0].shape[1]
        M = L
        slotb = np.zeros((1, M), np.int64)
        pcrowb = np.zeros((1, M), np.int64)
        mfb = np.zeros((1, M, Fd), np.float32)
        maskb = np.zeros((1, M), bool)
        psb = np.zeros((1, M), np.float32)
        thb = np.zeros((1, M), np.float32)
        ps2b = np.zeros((1, M), np.float32)
        vmb = np.zeros((1, M), bool)
        mfs, pss, ps2s, thrs, pci = rich[0]
        slotb[0, :L] = sv
        pcrowb[0, :L] = pci
        mfb[0, :L] = mfs
        maskb[0, :L] = True
        psb[0, :L] = pss
        thb[0, :L] = thrs
        ps2b[0, :L] = ps2s
        vmb[0, :L] = vm
        T, _, _, _ = fc.forward(base, fvb, slotb, pcrowb, mfb, maskb,
                                psb, thb, ps2b)
        Tv = T[0].detach().cpu().numpy()[:L]
        b = chess.Board(row["fen"])
        mus = [mv.uci() for mv in b.legal_moves]
        order = sorted(range(L), key=lambda j: -Tv[j])
        pick = mus[order[0]]
        margin = float(Tv[order[0]] - Tv[order[1]]) if L > 1 else 0.0
        ok = None
        if row.get("_pre") is not None:
            pres, ri = row["_pre"]
            off = int(pres["fen_off"][ri])
            o = pres["opt"][off:off + L] > 0.5
            ok = bool(o[order[0]])
        return {"pick": pick, "margin": margin, "correct": ok,
                "mus": mus, "order": order, "Tv": Tv.tolist()}

    # sample questions from the union of member fens
    rng = random.Random(0)
    sample_fens = rng.sample(fen_set, min(SAMPLE, len(fen_set)))
    home = {}
    for fi, f in enumerate(flies):
        for _, fen in f["qs"]:
            home[fen] = fi

    results = []
    t0 = time.time()
    for qi, fen in enumerate(sample_fens):
        pool = None
        for pl in ("DEGM2_Ch1", "DEGM2_Ch2", "DEGM2_Ch3", "DEGM2_Ch4",
                   "DEGM2_Ch5", "DEGM2_Ch6", "DEGM2_Ch7", "DEGM2_Ch8",
                   "DEGM2_Ch9", "DEGM2_Ch10", "DEGM2_Ch11", "DEGM2_Ch12",
                   "DEGM2_Ch13", "DEGM2_Ch14", "DEGM2_Ch15"):
            try:
                r = None
                pool_cache.setdefault(pl, {})
                if fen in pool_cache[pl]:
                    r = pool_cache[pl][fen]
                    pool = pl
                    break
            except Exception:
                pass
        if r is None:
            for pl in ("DEGM2_Ch1", "DEGM2_Ch2", "DEGM2_Ch3",
                       "DEGM2_Ch4", "DEGM2_Ch5", "DEGM2_Ch6",
                       "DEGM2_Ch7", "DEGM2_Ch8", "DEGM2_Ch9",
                       "DEGM2_Ch10", "DEGM2_Ch11", "DEGM2_Ch12",
                       "DEGM2_Ch13", "DEGM2_Ch14", "DEGM2_Ch15"):
                cache = {x["fen"]: x
                         for x in fc.load_pools([pl])}
                pool_cache[pl] = cache
                if fen in cache:
                    pool, r = pl, cache[fen]
                    break
        row = r
        qf = F[fen_idx[fen]]
        qrec = {"fen": fen, "pool": row["pool"], "home_fly":
                home.get(fen), "flies": []}
        sims = F @ qf
        answers = {}
        for fi, f in enumerate(flies):
            qr = query_fly(f, row)
            qr["fly"] = fi
            qr["knn"] = round(float(sims[f["midx"]].max()), 4)
            answers.setdefault(qr["pick"], []).append(fi)
            qrec["flies"].append(qr)
        modal = max(answers.items(), key=lambda kv: len(kv[1]))
        for qr in qrec["flies"]:
            qr["agree"] = len(answers.get(qr["pick"], []))
            qr["modal_pick"] = qr["pick"] == modal[0]
        n_correct = sum(1 for qr in qrec["flies"]
                        if qr["correct"])
        qrec["n_correct"] = n_correct
        qrec["modal_correct"] = bool(
            [qr["correct"] for qr in qrec["flies"]
             if qr["pick"] == modal[0] and qr["correct"] is not None])
        results.append(qrec)
        if (qi + 1) % 5 == 0:
            print(json.dumps({"q": qi + 1, "n_correct": n_correct,
                              "elapsed_s": round(time.time() - t0)}),
                  flush=True)
    json.dump(results, open(OUTP, "w"))
    print(json.dumps({"sampled": len(results),
                      "mean_n_correct": round(float(np.mean(
                          [q["n_correct"] for q in results])), 2),
                      "oracle_coverage": round(float(np.mean(
                          [q["n_correct"] >= 1 for q in results])), 3)}),
          flush=True)
    print("MOF-ROUTING-DATA-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
