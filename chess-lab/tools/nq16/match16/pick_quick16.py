#!/usr/bin/env python3
"""Pick best net per expert across runs-10min / runs-extra10 by val_corr
-> engine16quick/<expert>.nnue"""
import csv
import glob
import os
import shutil

RUNS = ["/srv/workspace/chess-active/runs-10min", "/srv/workspace/chess-active/runs-extra10"]
OUT = "/srv/workspace/chess-active/engine16quick"
EXPERTS = ["tb", "mvr", "rv2m", "qvmat", "nvb", "piece_down", "oppb", "dv_Q",
           "dv_R", "dv_rest", "op_pawnimb", "op_even_l0", "op_even_l1",
           "op_even_l2p", "mg_unsafe", "mg_safe"]
os.makedirs(OUT, exist_ok=True)
for e in EXPERTS:
    best = (None, -1)
    for R in RUNS:
        fs = sorted(glob.glob(f"{R}/{e}/lightning_logs/version_*/metrics.csv"),
                    key=os.path.getmtime)
        if not fs:
            continue
        for r in csv.DictReader(open(fs[-1])):
            c = r.get("val_corr")
            if c in (None, ""):
                continue
            try:
                v = float(c)
            except ValueError:
                continue
            if v > best[1]:
                best = (f"{R}/{e}/final.nnue", v)
    if best[0]:
        shutil.copy(best[0], f"{OUT}/{e}.nnue")
        print(f"{e:<12} corr={best[1]:.4f}  <- {best[0].split('/')[-2]}")
    else:
        print(f"{e:<12} MISSING")
