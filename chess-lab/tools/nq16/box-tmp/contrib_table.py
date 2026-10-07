import json
c = json.load(open("/extnvme/active/contrib_main.json"))["contrib"]
cap = json.load(open("/mnt/cold-raid6/chess-audit/train23/cap50_plan.json"))
hdr = f"{'expert':20s} {'decisive':>9s} {'draws':>9s} {'puzzles':>9s} {'DEGM':>6s} {'syzygy':>7s} {'aug':>4s}"
print(hdr)
td = tw = tp = 0
for e in sorted(cap, key=lambda x: -sum(cap[x][:2])):
    d = c.get(e, {}).get("decisive", 0)
    w = c.get(e, {}).get("draws", 0)
    p = cap[e][1]
    td += d; tw += w; tp += p
    syz = "12,547" if e == "tb" else "0"
    print(f"{e:20s} {d:9,d} {w:9,d} {p:9,d} {'0':>6s} {syz:>7s} {'0':>4s}")
print(f"{'TOTAL':20s} {td:9,d} {tw:9,d} {tp:9,d} {'0':>6s} {'12,547':>7s} {'0':>4s}")
