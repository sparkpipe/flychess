"""Single-fly step-rate benchmark: measures tb_step sps on this node's
GPU (fresh tiny fly, DEGM_Ch4_s1 pool). Run identically on rtx5090 and
a spark for the training-time comparison."""
import sys
import os
import random
import time
import traceback

sys.path.insert(0, os.path.expanduser("~/chess-lab"))
os.environ.setdefault("ANORM", "1")
import chess
import torch
import fly_curriculum as fc
import flyfeat_cb

flyfeat_cb.feat_vec(chess.Board())
rows = fc.load_pools(["DEGM_Ch4_s1"])
rmap = fc.build_retino_map(mode="geo")
m = fc.FlyCB(len(flyfeat_cb.FEATURE_KEYS), sel_boards=None,
             readout="variance").to(fc.DEV)
m.retino = rmap
m.retino_gain = torch.nn.Parameter(torch.ones(7) * 2.0).to(fc.DEV)
opt = torch.optim.Adam(m.parameters(), lr=3e-4)
rng = random.Random(1)
# warmup
for _ in range(10):
    fc.tb_step(m, opt, rows, rng)
torch.cuda.synchronize()
t0 = time.time()
N = 100
for _ in range(N):
    fc.tb_step(m, opt, rows, rng)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"BENCH {torch.cuda.get_device_name(0)}: {N/dt:.2f} tb_steps/s "
      f"({dt/N*1000:.0f} ms/step, batch 32)")
