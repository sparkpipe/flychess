import csv, glob, os

for r in sorted(glob.glob(os.path.expanduser("~/runanti_*"))):
    e = os.path.basename(r)[8:]
    mfiles = sorted(glob.glob(r + "/lightning_logs/version_*/metrics.csv"),
                    key=os.path.getmtime)
    if not mfiles:
        continue
    best = None
    with open(mfiles[-1]) as fh:
        for row in csv.DictReader(fh):
            c = row.get("val_corr")
            if c in (None, ""):
                continue
            try:
                t = (int(row["epoch"]), float(c))
            except (ValueError, KeyError):
                continue
            if best is None or t[1] > best[1]:
                best = t
    if best:
        print(f"{e} {best[0]} {best[1]:.4f}")
