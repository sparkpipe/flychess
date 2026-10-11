"""CORR AT EVERY CHECKPOINT (e19,39,...,319) per expert, plus bin sizes."""
import os, glob

runs = []
cur_name = None
cur_rows = []
lines = open("/tmp/named_all2.csv", errors="replace").readlines()
i = 0
while i < len(lines):
    ln = lines[i]
    if ln.startswith("RUN "):
        if cur_name:
            runs.append((cur_name, cur_rows))
        cur_name = ln[4:].strip()
        cur_rows = []
        i += 1
        if i < len(lines) and lines[i].startswith("epoch"):
            i += 1
        continue
    cur_rows.append(ln)
    i += 1
if cur_name:
    runs.append((cur_name, cur_rows))

CK = [19, 39, 59, 79, 99, 119, 139, 159, 179, 199, 219, 239, 259, 279, 299, 319]
best = {}
for name, rows in runs:
    per_ep = {}
    last_ep = -1
    for r in rows:
        p = r.rstrip("\n").split(",")
        if len(p) < 8:
            continue
        try:
            ep = int(float(p[0])) if p[0] else None
            corr = float(p[7]) if p[7] else None
        except (ValueError, IndexError):
            continue
        if ep is not None:
            last_ep = max(last_ep, ep)
            if corr is not None:
                per_ep[ep] = corr
    if last_ep < 0:
        continue
    if name not in best or last_ep > best[name][0]:
        best[name] = (last_ep, per_ep)

hdr = "%-28s" % "engine/expert" + "".join("%6s" % ("e%d" % e) for e in CK)
print(hdr)
for name in sorted(best):
    ep, per_ep = best[name]
    row = "%-28s" % name
    for e in CK:
        v = per_ep.get(e)
        row += ("%6.2f" % v) if v is not None else "%6s" % "-"
    print(row)

print("\ntrain-bin sizes (records):")
for eng in ("main", "anti"):
    for b in sorted(glob.glob(f"/mnt/cold-raid6/chess-audit/train23/{eng}/*.train.bin")):
        n = os.path.getsize(b) // 40
        print("%-30s %9s" % (eng + "/" + os.path.basename(b)[:-10], format(n, ",")))
