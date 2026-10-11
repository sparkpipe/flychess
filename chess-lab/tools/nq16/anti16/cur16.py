#!/usr/bin/env python3
"""Per-expert val_corr trajectory summary across run16_* dirs."""
import csv, glob, os

runs = sorted(glob.glob(os.path.expanduser("~/run16_*")))
if not runs:
    runs = ["/srv/workspace/chess-active/run16_op_even_l1"]
for r in runs:
    exp = os.path.basename(r).replace("run16_", "")
    mfiles = sorted(glob.glob(r + "/lightning_logs/version_*/metrics.csv"),
                    key=os.path.getmtime)
    if not mfiles:
        print(f"{exp}: NO metrics")
        continue
    rows = []
    with open(mfiles[-1]) as fh:
        for row in csv.DictReader(fh):
            c = row.get("val_corr")
            if c not in (None, ""):
                try:
                    rows.append((int(row["step"]), float(c)))
                except (ValueError, KeyError):
                    pass
    if not rows:
        print(f"{exp}: no val rows")
        continue
    rows.sort()
    n = len(rows)
    corrs = [c for _, c in rows]
    bi = max(range(n), key=lambda i: corrs[i])

    def wbest(a, b):
        seg = corrs[a:b]
        return max(seg) if seg else float("nan")

    last15 = wbest(max(0, n - 15), n)
    prev30 = wbest(max(0, n - 45), max(0, n - 15))
    tail = " ".join(f"{c:.3f}" for c in corrs[-15:])
    print(f"{exp}: epochs={n} last_ep={rows[-1][0]} best={corrs[bi]:.4f}@ep{rows[bi][0]} "
          f"best15={last15:.4f} prev30={prev30:.4f} delta={last15 - prev30:+.4f} tail={tail}")
