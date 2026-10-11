"""Median/mean AE + corr for the oppb test net against bin labels."""
import sys, os, math
import torch

NNUE = "/srv/workspace/flychess/src/nnue-pytorch"
sys.path.insert(0, NNUE)
os.chdir(NNUE)

import model as M
import data_loader
from model.config import NNUELightningConfig

CK = "/extnvme/active/train16_test_oppb/lightning_logs/version_0/checkpoints/last.ckpt"
VAL = "/extnvme/active/train16/oppb.val.bin"

state = torch.load(CK, map_location="cpu", weights_only=False)
print("checkpoint epoch:", state.get("epoch"))

cfg = NNUELightningConfig(features="Full_Threats+PP_3Wide+HalfKAv2_hm^")
net = M.NNUE(cfg)
net.load_state_dict(state["state_dict"])
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
    "Full_Threats+PP_3Wide+HalfKAv2_hm", [VAL], 8192, config=skip,
    ddp_config=DataloaderDDPConfig(rank=0, world_size=1),
)
from torch.utils.data import DataLoader
dl = DataLoader(data_loader.FixedNumBatchesDataset(val_infinite, 4), batch_size=None)

d = cfg.loss_params.in_scaling
o = cfg.loss_params.in_offset
errs, xs, ys = [], [], []
with torch.no_grad():
    for batch in dl:
        us, them, wi, bi, outcome, score, pc = batch
        scorenet = net.model(us, them, wi, bi, pc,
                             cfg.use_fake_act_quantization, cfg.use_fake_weight_quantization)
        pred = scorenet * net.model.quantization.nnue2score
        true_cp = score  # bin scores are already cp
        pred_cp = pred
        errs.extend((pred_cp - true_cp).abs().tolist())
        xs.extend(true_cp.tolist())
        ys.extend(pred_cp.tolist())

errs = [abs(float(e)) for x in errs for e in (x if isinstance(x, list) else [x])]
xs = [float(x) for v in xs for x in (v if isinstance(v, list) else [v])]
ys = [float(y) for v in ys for y in (v if isinstance(v, list) else [v])]
n = len(errs)
errs.sort()
print(f"n            = {n:,}")
print(f"MEAN AE      = {sum(errs)/n:.2f} cp")
print(f"MEDIAN AE    = {errs[n//2]:.2f} cp")
print(f"p90 AE       = {errs[int(n*0.9)]:.2f} cp")
print(f"p99 AE       = {errs[int(n*0.99)]:.2f} cp")
mx, my = sum(xs)/n, sum(ys)/n
cov = sum((a-mx)*(b-my) for a, b in zip(xs, ys))/n
vx = sum((a-mx)**2 for a in xs)/n
vy = sum((b-my)**2 for b in ys)/n
print(f"corr         = {cov/math.sqrt(max(vx*vy,1e-12)):+.4f}")
