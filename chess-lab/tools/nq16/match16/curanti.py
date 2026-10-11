import csv, glob, os

for r in sorted(glob.glob(os.path.expanduser("~/runanti_*"))):
    e = os.path.basename(r)[8:]
    mfiles = sorted(glob.glob(r + "/lightning_logs/version_*/metrics.csv"),
                    key=os.path.getmtime)
    if not mfiles:
        print(f"{e}: no metrics")
        continue
    rows = []
    with open(mfiles[-1]) as fh:
        for row in csv.DictReader(fh):
            c = row.get("val_corr")
            if c in (None, ""):
                continue
            try:
                rows.append((int(row["epoch"]), float(c)))
            except (ValueError, KeyError):
                continue
    if not rows:
        print(f"{e}: no val rows")
        continue
    rows.sort()
    n = len(rows)
    corrs = [c for _, c in rows]
    bi = max(range(n), key=lambda i: corrs[i])

    def wbest(a, b):
        seg = corrs[a:b]
        return max(seg) if seg else float("nan")

    last10 = wbest(max(0, n - 10), n)
    prev20 = wbest(max(0, n - 30), max(0, n - 10))
    print(f"{e}: epochs={n} best={corrs[bi]:.4f}@e{rows[bi][0]} "
          f"best10={last10:.4f} prev20={prev20:.4f} delta={last10 - prev20:+.4f}")
# d20 job progress
for j in glob.glob(os.path.expanduser("~/d20job.txt")):
    o = os.path.expanduser("~/d20out.tsv")
    jn = sum(1 for _ in open(j))
    on = sum(1 for _ in open(o)) if os.path.exists(o) else 0
    print(f"d20: done {on} of {jn} (remaining {max(0, jn - on)})")
