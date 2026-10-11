"""N single-question flies trained CONCURRENTLY in one process.

Answers the operator's two questions about the serial harness:
- batch waste: a 1-row pool filled the training batch (B=256) with 256
  copies of the SAME position; every loss in tb_step is mean-reduced,
  so that gradient is IDENTICAL to batch-of-1 — 255/256 of the compute
  bought nothing. Here B=1 (via env before importing fly_curriculum).
- vector size: solution vectors carry TRAINABLE deltas only. The old
  296MB vec files were the frozen WT connectome buffer riding along in
  state_dict with zero delta; the trainable surface is what moves and
  what any combination experiment actually mixes.

Deterministic question picks (crc32 seeds — the old hash() picks were
salted per process, so every relaunch trained different questions).

Usage: python3 tools/ten_parallel.py [N]     (default 10)
Outputs: ten_q/vec_{qi}.pt + ten_q/report.json
"""
import sys
import os
import json
import copy
import random
import zlib

sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ANORM", "1")
os.environ.setdefault("B", "1")
import chess
import torch
import numpy as np
import fly_curriculum as fc
import flyfeat_cb
from fly_curriculum import DEV, load_pools, FAM_SCORE
from ten_questions import pack_pre

OUT = os.environ.get("TENQ_OUT", "/home/spec/chess-lab/ten_q")
GATE_EVERY = 10
STABLE_STREAK = 10
CAP = int(os.environ.get("CAP", "2000"))
LR = float(os.environ.get("LR", "3e-4"))


def build_model(seed=0):
    torch.manual_seed(seed)
    rmap = fc.build_retino_map(mode="geo")
    m = fc.FlyCB(len(flyfeat_cb.FEATURE_KEYS), sel_boards=None,
                 readout="variance").to(DEV)
    m.retino = rmap
    m.retino_gain = torch.nn.Parameter(torch.ones(7) * 2.0).to(DEV)
    return m


def pick_questions(n):
    chapters = [f"DEGM_Ch{i}" for i in (1, 4, 5, 8, 12)]
    qs = []
    for ch in chapters:
        rows = load_pools([ch])
        rng = random.Random(zlib.crc32(ch.encode()))
        qs += rng.sample(rows, 2)
    return qs[:n]


def main():
    flyfeat_cb.feat_vec(chess.Board())
    os.makedirs(OUT, exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    questions = pick_questions(n)

    # all pools + pre arrays up front — no mid-run file writes
    rows_by_q = {}
    for qi, e in enumerate(questions):
        pool = f"TENQ_{qi}"
        e = {k: v for k, v in e.items() if k != "_pre"}
        with open(f"/home/spec/chess-lab/tbpools/{pool}.jsonl", "w") as f:
            f.write(json.dumps(dict(e, pool=pool)) + "\n")
        pack_pre(pool, e)
        rows_by_q[qi] = load_pools([pool])
        FAM_SCORE[pool] = 0.0

    m0 = build_model(0)
    base_sd = {k: p.detach().cpu().clone() for k, p in m0.named_parameters()}
    n_tr = sum(p.numel() for p in m0.parameters())
    print(json.dumps({"init": "seed0-shared", "trainable_params": n_tr,
                      "vec_bytes_each": n_tr * 4, "B": fc.B}), flush=True)

    models = [build_model(0) for _ in questions]      # seed-0: identical
    for m in models[1:]:                              # share frozen WT
        m.WT = models[0].WT
    opts = [torch.optim.Adam(m.parameters(), lr=LR) for m in models]
    rngs = [random.Random(qi) for qi in range(len(questions))]
    mstate = [{"step": 0, "streak": 0, "first": None, "s0": None,
               "done": False, "pair": 0.0} for _ in questions]

    import time
    t0 = time.time()
    while any(not s["done"] for s in mstate) and \
            max(s["step"] for s in mstate) < CAP:
        for qi in range(len(questions)):
            s = mstate[qi]
            if s["done"]:
                continue
            if s["step"] % GATE_EVERY == 0:
                pair, _, _ = fc.gate_tb(models[qi], rows_by_q[qi],
                                        random.Random(777),
                                        exhaustive=True)
                s["pair"] = pair
                if s["step"] == 0:
                    s["s0"] = bool(pair >= 0.98)
                s["streak"] = s["streak"] + 1 if pair >= 0.98 else 0
                if s["first"] is None and pair >= 0.98:
                    s["first"] = s["step"]
                if s["streak"] >= STABLE_STREAK:
                    s["done"] = True
                    print(json.dumps({"q": qi, "stable": s["step"],
                                      "t": round(time.time() - t0)}),
                          flush=True)
                    continue
            fc.tb_step(models[qi], opts[qi], rows_by_q[qi], rngs[qi])
            s["step"] += 1

    report = []
    for qi, e in enumerate(questions):
        s = mstate[qi]
        d = {k: (p.detach().cpu() - base_sd[k])
             for k, p in models[qi].named_parameters()}
        torch.save(d, f"{OUT}/vec_{qi}.pt")
        mx = max(float(v.abs().max()) for v in d.values())
        eff = sum(int((v.abs() >= 0.01 * mx).sum()) for v in d.values())
        rec = {"q": qi, "fen": e["fen"], "cat": e["cat"], "best": e["best"],
               "n_moves": len(list(chess.Board(e["fen"]).legal_moves)),
               "step0_pass": s["s0"], "first_pass_step": s["first"],
               "stable_step": s["step"] if s["done"] else None,
               "final_pair": round(s["pair"], 3),
               "max_delta": round(mx, 6), "eff_support_1pct": eff}
        report.append(rec)
        print(json.dumps(rec), flush=True)

    with open(f"{OUT}/report.json", "w") as f:
        json.dump(report, f, indent=1)
    fps = [r["first_pass_step"] for r in report
           if r["first_pass_step"] is not None]
    print(json.dumps({"TEN-PARALLEL-COMPLETE": True,
                      "solved": len(fps), "of": len(report),
                      "first_pass": fps,
                      "mean_first_pass": sum(fps) / max(len(fps), 1),
                      "wall_s": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
