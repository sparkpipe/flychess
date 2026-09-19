"""MoF layer-0 worker: ONE dedicated fly for ONE DEGM chapter, trained to
the 98% EXHAUSTIVE gate, then frozen (muscle memory). Full engine reuse
(fly_curriculum's tb_step/gate_tb); own checkpoint per chapter.
Env: CH=<chapter>, resumes idempotently from flies_mof/l0_ch{CH}.pt.
"""
import os
import sys
import json
import random
import numpy as np
import torch
import chess

sys.path.insert(0, "/home/spec/chess-lab")
import fly_curriculum as fc
import flyfeat_cb
from fly_curriculum import DEV, load_pools, FAM_SCORE

CH = int(os.environ.get("CH", "1"))
STATE = f"/home/spec/chess-lab/flies_mof/l0_ch{CH}.pt"
CAP_STEPS = int(os.environ.get("CAP_STEPS", "40000"))


def main():
    torch.manual_seed(0)
    flyfeat_cb.feat_vec(chess.Board())
    rmap = fc.build_retino_map(mode="geo")
    m = fc.FlyCB(len(flyfeat_cb.FEATURE_KEYS), sel_boards=None,
                 readout="variance").to(DEV)
    m.retino = rmap
    m.retino_gain = torch.nn.Parameter(torch.ones(7) * 2.0).to(DEV)
    if os.path.exists(STATE):
        m.load_state_dict(torch.load(STATE, weights_only=True),
                          strict=False)
        print(f"ch{CH}: resumed", flush=True)
    else:
        print(f"ch{CH}: FRESH dedicated fly", flush=True)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    rows = load_pools([f"DEGM_Ch{CH}"])
    rng = random.Random(6000 + CH)
    FAM_SCORE.update({f"DEGM_Ch{CH}": 0.0})
    step = 0
    while step < CAP_STEPS:
        for _ in range(500):
            step += 1
            fc.tb_step(m, opt, rows, rng)
        torch.save(m.state_dict(), STATE + ".tmp")
        os.replace(STATE + ".tmp", STATE)
        pair, _, fam = fc.gate_tb(m, rows, random.Random(777),
                                  exhaustive=True)
        worst = min(fam.values()) if fam else 0.0
        FAM_SCORE.update(fam)
        print(json.dumps({"l0_ch": CH, "step": step,
                          "exhaustive": round(pair, 4)}), flush=True)
        if pair >= 0.98:
            print(f"L0-CH{CH}-PASSED at step {step} "
                  f"(exhaustive {pair:.4f})", flush=True)
            break
    else:
        print(f"L0-CH{CH}-CAP-EXHAUSTED (last {pair:.4f})", flush=True)


if __name__ == "__main__":
    main()
