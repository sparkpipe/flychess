"""Dump weight vectors from exported .nnue levels via serialize's own path.

Usage (from /srv/workspace/flychess/src/nnue-pytorch):
  python3 dump_wvec.py <config_pickle_args...> — no; simplest: reuse the
  train.py config parser with the same feature set the training used.
Loads each .nnue with M.NNUEReader using the training config, flattens
parameters, writes /extnvme/traj/<expert>-e<ep>.npy
"""
import sys, glob, os, re
import numpy as np
import torch

sys.path.insert(0, "/srv/workspace/flychess/src/nnue-pytorch")
import model as M
from model.config import ModelConfig
from features import get_feature_set_from_name

fs = get_feature_set_from_name("Full_Threats+PP_3Wide+HalfKAv2_hm^")
mc = ModelConfig()

NETS = "/mnt/cold-raid6/chess-audit/runs23/anti/nets"
OUT = "/extnvme/traj"
os.makedirs(OUT, exist_ok=True)
for f in sorted(glob.glob(f"{NETS}/*-e*.nnue")):
    m = re.match(r"(.+)-e(\d+)\.nnue", os.path.basename(f))
    if not m:
        continue
    out = f"{OUT}/{m.group(1)}-e{m.group(2)}.npy"
    if os.path.exists(out):
        continue
    with open(f, "rb") as fh:
        reader = M.NNUEReader(fh, fs, config=mc)
        model = reader.model
    vec = np.concatenate([p.detach().numpy().ravel().astype(np.float32)
                          for p in model.parameters()])
    np.save(out, vec)
    print("dumped", m.group(1), "e" + m.group(2), f"{len(vec):,} params", flush=True)
print("ALL DUMPED")
