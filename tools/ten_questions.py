"""Ten single-question flies (operator directive): train 10 separate
one-question flies on 10 different DEGM questions, record for each:
step-0 pass (already-knew vs learned), first-pass step, stable step,
and SAVE the trained-delta vector (weights relative to the shared fresh
init) as the solution vector for combination experiments.

Usage: python3 tools/ten_questions.py
Outputs: ten_q/vec_{i}.pt (delta state_dicts), ten_q/report.json
"""
import sys
import os
import json
import random
import glob

sys.path.insert(0, "/home/spec/chess-lab")
os.environ.setdefault("ANORM", "1")
import chess
import torch
import numpy as np
import fly_curriculum as fc
import flyfeat_cb
from fly_curriculum import DEV, load_pools, FAM_SCORE

OUT = "/home/spec/chess-lab/ten_q"
GATE_EVERY = 10          # fine-grained: catch the true first-pass step
STABLE_STREAK = 10
CAP = 2000
LR = float(os.environ.get("LR", "3e-4"))


def build_model(seed=0):
    torch.manual_seed(seed)
    rmap = fc.build_retino_map(mode="geo")
    m = fc.FlyCB(len(flyfeat_cb.FEATURE_KEYS), sel_boards=None,
                 readout="variance").to(DEV)
    m.retino = rmap
    m.retino_gain = torch.nn.Parameter(torch.ones(7) * 2.0).to(DEV)
    return m


def delta(m, base_sd):
    """Solution vector: trained weights minus the shared init."""
    out = {}
    for k, v in m.state_dict().items():
        if v.dtype.is_floating_point:
            out[k] = (v.detach().cpu() - base_sd[k].cpu())
    return out


def train_one(qi, e):
    pool = f"TENQ_{qi}"
    with open(f"/home/spec/chess-lab/tbpools/{pool}.jsonl", "w") as f:
        f.write(json.dumps(dict({k: v for k, v in e.items()
                                 if k != "_pre"},
                                pool=pool)) + "\n")
    rows = load_pools([pool])
    m = build_model(0)               # SHARED init seed — deltas comparable
    base_sd = {k: v.detach().cpu().clone()
               for k, v in m.state_dict().items()}
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    rng = random.Random(qi)
    FAM_SCORE.update({pool: 0.0})
    step = 0
    streak = 0
    first_pass = None
    step0_pass = None
    while step < CAP:
        # gate BEFORE any training: the step-0 read (already-knew?)
        pair, _, _ = fc.gate_tb(m, rows, random.Random(777),
                                exhaustive=True)
        if step == 0:
            step0_pass = bool(pair >= 0.98)
        streak = streak + 1 if pair >= 0.98 else 0
        if first_pass is None and pair >= 0.98:
            first_pass = step
        if streak >= STABLE_STREAK:
            break
        for _ in range(GATE_EVERY):
            step += 1
            fc.tb_step(m, opt, rows, rng)
    rec = {"q": qi, "fen": e["fen"], "cat": e["cat"],
           "best": e["best"],
           "n_moves": len(list(chess.Board(e["fen"]).legal_moves)),
           "step0_pass": step0_pass,
           "first_pass_step": first_pass,
           "stable_step": step if streak >= STABLE_STREAK else None,
           "final_pair": round(pair, 3)}
    torch.save(delta(m, base_sd), f"{OUT}/vec_{qi}.pt")
    return rec


def main():
    flyfeat_cb.feat_vec(chess.Board())
    os.makedirs(OUT, exist_ok=True)
    # 10 diverse questions: 2 per chapter from ch1..ch15 spread
    chapters = [f"DEGM_Ch{i}" for i in (1, 4, 5, 8, 12)]
    questions = []
    for ch in chapters:
        rows = load_pools([ch])
        rng = random.Random(hash(ch) % 99991)
        picks = rng.sample(rows, 2)
        questions += picks
    report = []
    for qi, e in enumerate(questions):
        rec = train_one(qi, e)
        report.append(rec)
        print(json.dumps(rec), flush=True)
    with open(f"{OUT}/report.json", "w") as f:
        json.dump(report, f, indent=1)
    fps = [r["first_pass_step"] for r in report if
           r["first_pass_step"] is not None]
    s0 = sum(1 for r in report if r["step0_pass"])
    print(json.dumps({"TEN-QUESTIONS-COMPLETE": True,
                      "step0_pass": s0, "of": len(report),
                      "first_pass": fps,
                      "mean_first_pass":
                          sum(fps) / max(len(fps), 1)}), flush=True)


if __name__ == "__main__":
    main()
