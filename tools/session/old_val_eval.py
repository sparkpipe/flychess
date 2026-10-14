#!/usr/bin/env python3
"""Evaluate a fleet-run best ckpt against the OLD (bins-db-v1) val bin —
apples-to-apples corr vs the pre-fleet number."""
import glob
import math
import re
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/srv/workspace/flychess/src/nnue-pytorch")
import os
os.chdir("/srv/workspace/flychess/src/nnue-pytorch")
import data_loader
import model as M
from model.config import NNUELightningConfig

expert = sys.argv[1]
feat = "Full_Threats+PP_3Wide+HalfKAv2_hm^"
cfg = NNUELightningConfig(features=feat)
net = M.NNUE(cfg)
cks = sorted(glob.glob(f"/srv/workspace/chess-active/runs-fleet1h/{expert}/"
                       f"lightning_logs/version_*/checkpoints/epoch=*.ckpt"),
             key=lambda f: int(re.search(r"epoch=(\d+)", f).group(1)))
st = torch.load(cks[-1], map_location="cpu", weights_only=False)
net.load_state_dict(st["state_dict"])
net.eval()
from types import SimpleNamespace as NS
from data_loader.config import DataloaderDDPConfig
skip = NS(filtered=False, random_fen_skipping=0, wld_filtered=False,
          early_fen_skipping=False, soft_early_fen_skipping=False,
          simple_eval_skipping=False, param_index=0,
          pc_y0=0.0, pc_y1=0.0, pc_y2=0.0, pc_y3=0.0, pc_y4=0.0,
          ply_x1=0.0, ply_y1=0.0, ply_x2=0.0, ply_y2=0.0,
          ply_x3=0.0, ply_y3=0.0, ply_x4=0.0, ply_y4=0.0)
val_infinite = data_loader.SparseBatchDataset(
    feat.rstrip("^"), [f"/srv/workspace/chess-active/bins-db-v1/{expert}.val.bin"], 8192,
    config=skip, ddp_config=DataloaderDDPConfig(rank=0, world_size=1))
dl = DataLoader(data_loader.FixedNumBatchesDataset(val_infinite, 4), batch_size=None)
xs, ys = [], []
with torch.no_grad():
    for batch in dl:
        us, them, wi, bi, outcome, score, pc = batch
        pred = net(us, them, wi, bi, pc,
                   cfg.use_fake_act_quantization, cfg.use_fake_weight_quantization)
        xs.extend(score.flatten().tolist())
        ys.extend(pred.flatten().tolist())
n = len(xs)
mx, my = sum(xs) / n, sum(ys) / n
cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / n
vx = sum((a - mx) ** 2 for a in xs) / n
vy = sum((b - my) ** 2 for b in ys) / n
print(f"fleet-{expert} net on OLD {expert} val (n={n}): corr={cov/math.sqrt(vx*vy):+.4f}")
