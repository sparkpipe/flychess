"""ERR RATES — final val_mae/val_corr per completed expert from metrics.csv."""
import glob, csv, os

RUNS = "/mnt/cold-raid6/chess-audit/runs23"
for eng in ("main", "anti"):
    rows = []
    for d in sorted(glob.glob(f"{RUNS}/{eng}/*/lightning_logs/version_*/metrics.csv")):
        expert = d.split(f"runs23/{eng}/")[1].split("/")[0]
        try:
            with open(d) as f:
                r = list(csv.DictReader(f))
            last = None
            for row in r:
                if row.get("val_mae") not in (None, ""):
                    last = row
            if last:
                mae = float(last["val_mae"]) / 340.0
                corr = float(last.get("val_corr") or 0)
                ep = last.get("epoch", "?")
                rows.append((expert, ep, mae, corr))
        except Exception:
            pass
    if rows:
        print(f"== {eng} ==")
        print(f"{'expert':>22} {'ep':>4} {'MAE(cp)':>8} {'corr':>6}")
        for e, ep, m, c in sorted(rows, key=lambda x: x[2]):
            print(f"{e:>22} {ep:>4} {m:>8.1f} {c:>6.3f}")
