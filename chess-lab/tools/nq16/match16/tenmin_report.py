#!/usr/bin/env python3
"""Per-expert best val_corr / val_mae from the 10-min runs + engine eval-sign verdict."""
import csv
import glob
import os

R = "/srv/workspace/chess-active/runs-10min"
print(f"{'expert':<14} {'epochs':>7} {'best_corr':>10} {'@ep':>5} {'mae@best':>9} {'final_mae':>10}")
for e in sorted(os.listdir(R)):
    fs = sorted(glob.glob(f"{R}/{e}/lightning_logs/version_*/metrics.csv"),
                key=os.path.getmtime)
    if not fs:
        continue
    rows = []
    for r in csv.DictReader(open(fs[-1])):
        c = r.get("val_corr")
        if c not in (None, ""):
            try:
                rows.append((int(r["epoch"]), float(c),
                             float(r["val_mae"]) if r.get("val_mae") else None))
            except (ValueError, KeyError):
                pass
    if not rows:
        print(f"{e:<14} no val rows")
        continue
    rows.sort()
    b = max(rows, key=lambda x: x[1])
    print(f"{e:<14} {len(rows):>7} {b[1]:>10.4f} {b[0]:>5} {b[2] or 0:>9.1f} {rows[-1][2] or 0:>10.1f}")
