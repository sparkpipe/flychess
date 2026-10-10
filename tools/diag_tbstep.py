import os, random, traceback
os.environ.setdefault("ANORM", "1")
import chess, torch
import fly_curriculum as fc, flyfeat_cb
flyfeat_cb.feat_vec(chess.Board())
rows = fc.load_pools(["DEGM_Ch4_s1"])
rmap = fc.build_retino_map(mode="geo")
m = fc.FlyCB(len(flyfeat_cb.FEATURE_KEYS), sel_boards=None,
             readout="variance").to(fc.DEV)
m.retino = rmap
m.retino_gain = torch.nn.Parameter(torch.ones(7) * 2.0).to(fc.DEV)
opt = torch.optim.Adam(m.parameters(), lr=3e-4)
rng = random.Random(1)
try:
    for _ in range(20):
        fc.tb_step(m, opt, rows, rng)
    print("TRAIN-PATH-OK")
except Exception:
    traceback.print_exc()
