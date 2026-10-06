import csv, glob, os

for r in sorted(glob.glob(os.path.expanduser("~/run16_*"))):
    e = os.path.basename(r)[6:]
    mfiles = sorted(glob.glob(r + "/lightning_logs/version_*/metrics.csv"),
                    key=os.path.getmtime)
    if not mfiles:
        print(f"{e}: no metrics")
        continue
    best = None
    with open(mfiles[-1]) as fh:
        for row in csv.DictReader(fh):
            c = row.get("val_corr")
            if c in (None, ""):
                continue
            try:
                ep, v = int(row["epoch"]), float(c)
            except (ValueError, KeyError):
                continue
            if best is None or v > best[1]:
                best = (ep, v)
    if not best:
        print(f"{e}: no val rows")
        continue
    ep, v = best
    net = f"{r}/nets/{e}_e{ep}.nnue"
    print(f"{e}: best_epoch={ep} corr={v:.4f} net={'OK' if os.path.exists(net) else 'MISSING'}")
