import sys, json
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import router12 as R

c = json.load(open("/extnvme/active/contrib_main.json"))["contrib"]

degm = {}
for path in ("/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/degm_full.tsv",):
    for line in open(path, errors="ignore"):
        p = line.rstrip("\n").split("|")
        if not p or not p[0].strip():
            continue
        e = R.route23(p[0])
        degm[e] = degm.get(e, 0) + 1

syz = {}
for line in open("/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/tb.tsv", errors="ignore"):
    p = line.rstrip("\n").split("|")
    if not p or not p[0].strip():
        continue
    e = R.route23(p[0])
    syz[e] = syz.get(e, 0) + 1

cap = json.load(open("/mnt/cold-raid6/chess-audit/train23/cap50_plan.json"))
td = tw = tg = ts = 0
print(f"{'expert':20s} {'decisive':>9s} {'draws':>9s} {'DEGM':>7s} {'syzygy':>7s} {'puzzles':>8s} {'TOTAL-B':>9s}")
for e in sorted(cap, key=lambda x: -(c.get(x, {}).get("decisive", 0) + c.get(x, {}).get("draws", 0))):
    d = c.get(e, {}).get("decisive", 0)
    w = c.get(e, {}).get("draws", 0)
    g = degm.get(e, 0)
    s = syz.get(e, 0)
    tot = d + w + g + s
    td += d; tw += w; tg += g; ts += s
    print(f"{e:20s} {d:9,d} {w:9,d} {g:7,d} {s:7,d} {'0':>8s} {tot:9,d}")
print(f"{'TOTAL':20s} {td:9,d} {tw:9,d} {tg:7,d} {ts:7,d} {'0':>8s} {td+tw+tg+ts:9,d}")
