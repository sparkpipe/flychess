"""ONE-question training (operator): a dedicated fly learns a single
DEGM position to the 98% exhaustive gate (i.e. picks the optimal move
on that one board, held across consecutive gates). Reports steps taken
and total board-visits — the per-question iteration number, measured.
"""
import sys
import os
import json
import random

sys.path.insert(0, "/home/spec/chess-lab")
os.environ.setdefault("ANORM", "1")
import chess
import torch
import fly_curriculum as fc
import flyfeat_cb

flyfeat_cb.feat_vec(chess.Board())
e = json.loads(open("/tmp/one_question.jsonl").read())
rows = [dict(e, pool="ONE_Q")]

rmap = fc.build_retino_map(mode="geo")
m = fc.FlyCB(len(flyfeat_cb.FEATURE_KEYS), sel_boards=None,
             readout="variance").to(fc.DEV)
m.retino = rmap
m.retino_gain = torch.nn.Parameter(torch.ones(7) * 2.0).to(fc.DEV)
opt = torch.optim.Adam(m.parameters(), lr=3e-4)
rng = random.Random(1)
fc.FAM_SCORE.update({"ONE_Q": 0.0})
step = 0
streak = 0
first_hit = None
while step < 20000 and streak < 10:      # 10 consecutive passing gates
    for _ in range(50):
        step += 1
        fc.tb_step(m, opt, rows, rng)
    pair, _, _ = fc.gate_tb(m, rows, random.Random(777), exhaustive=True)
    streak = streak + 1 if pair >= 0.98 else 0
    if first_hit is None and pair >= 0.98:
        first_hit = step
    print(json.dumps({"step": step, "hit": round(pair, 3),
                      "streak": streak}), flush=True)
print(json.dumps({"RESULT": "STABLE" if streak >= 10 else "CAP",
                  "first_pass_step": first_hit,
                  "total_steps": step,
                  "board_visits": step * 32}), flush=True)
