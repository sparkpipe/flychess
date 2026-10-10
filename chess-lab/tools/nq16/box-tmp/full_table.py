"""COMPLETE per-expert table: every trained run, last-epoch + per-checkpoint
MAE/median-proxy and corr. Checkpoint epochs = 19,59,99,159,239,319.
Also computes the same for the nQ bar runs (bar_run csv if present)."""

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

CK = (19, 59, 99, 159, 239, 319)
best = {}
for name, rows in runs:
    per_ep = {}
    last = None
    for r in rows:
        p = r.rstrip("\n").split(",")
        if len(p) < 9:
            continue
        try:
            ep = int(float(p[0])) if p[0] else None
            mae = float(p[9]) / 340.0 if len(p) > 9 and p[9] else None
            corr = float(p[7]) if len(p) > 7 and p[7] else None
        except (ValueError, IndexError):
            continue
        if ep is not None and mae is not None:
            last = (ep, mae, corr)
            per_ep[ep] = (mae, corr)
    if not last:
        continue
    # keep the latest-epoch record per name
    key = name
    if key not in best or last[0] > best[key][0]:
        best[key] = (last[0], per_ep)

print("%-30s %5s %8s %6s | checkpoint MAE(cp)/corr trail" % ("engine/expert", "ep", "MAE", "corr"))
for name in sorted(best):
    ep, per_ep = best[name]
    last = per_ep[ep]
    trail = "  ".join(
        "e%d:%.0f/%.2f" % (e, per_ep[e][0], per_ep[e][1] or 0)
        for e in CK if e in per_ep)
    print("%-30s %5d %8.1f %6.3f | %s" % (name, ep, last[0], last[1] or 0, trail))
print("total runs:", len(best))
