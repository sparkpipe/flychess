import csv

runs = []
cur_name = None
cur_rows = []
lines = open("/tmp/named_all.csv", errors="replace").readlines()
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

hdr = ("engine/expert", "lastEp", "MAE(cp)", "corr")
print("%-26s %6s %8s %6s   trajectory" % hdr)
for name, rows in runs:
    last = None
    traj = {}
    for r in rows:
        p = r.rstrip("\n").split(",")
        if len(p) < 9:
            continue
        try:
            ep = int(float(p[0])) if p[0] else None
            mae = float(p[8]) / 340.0 if p[8] else None
            corr = float(p[7]) if p[7] else None
        except (ValueError, IndexError):
            continue
        if ep is not None and mae is not None:
            last = (ep, mae, corr)
            if ep % 80 == 79 or ep in (19, 59):
                traj[ep] = mae
    if last:
        t = "  ".join("e%d:%.0f" % (e, m) for e, m in sorted(traj.items()))
        print("%-26s %6d %8.1f %6.3f   %s" % (name, last[0], last[1], last[2] or 0, t))
print("runs:", len(runs))
