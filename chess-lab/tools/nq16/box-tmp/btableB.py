"""Revised-taxonomy contribution table: every source routed through routerB."""
import sys, os
from collections import Counter
sys.path.insert(0, "/tmp")
from routerB import routeB

SEG = "/extnvme/segments"

ORDER = ["tb","mvr","rv2m","qvmat","nvb","piece_down","oppb","dv",
         "op_pawnimb","op_even_l0","op_even_l1","op_even_l2p",
         "mg_unsafe","mg_safe_sym","mg_safe_one_castled"]
contrib = {e: Counter() for e in ORDER}


n = 0
for fn in sorted(os.listdir(SEG)):
    if not (fn.startswith("pos_") and (fn.endswith(".clean.tsv") or fn.endswith(".draws.tsv"))):
        continue
    src = "decisive" if fn.endswith(".clean.tsv") else "draws"
    for line in open(f"{SEG}/{fn}", errors="ignore"):
        p = line.rstrip("\n").split("|")
        if len(p) < 6 or p[3] != "pos":
            continue
        contrib[routeB(p[0])][src] += 1
        n += 1
        if n >= 4000000:
            break
    if n >= 4000000:
        break

for path, name in [("/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/degm_full.tsv", "DEGM"),
                   ("/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/tb.tsv", "syzygy")]:
    for line in open(path, errors="ignore"):
        p = line.rstrip("\n").split("|")
        if not p or not p[0].strip():
            continue
        contrib[routeB(p[0])][name] += 1

td = tw = tg = ts = 0
print(f"{'#':>2} {'expert':22s} {'decisive':>9s} {'draws':>9s} {'DEGM':>7s} {'syzygy':>7s} {'TOTAL':>9s}")
for i, e in enumerate(ORDER, 1):
    d = contrib[e]["decisive"]; w = contrib[e]["draws"]
    g = contrib[e]["DEGM"]; s = contrib[e]["syzygy"]
    td += d; tw += w; tg += g; ts += s
    print(f"{i:>2} {e:22s} {d:9,d} {w:9,d} {g:7,d} {s:7,d} {d+w+g+s:9,d}")
print(f"{'':2} {'TOTAL':22s} {td:9,d} {tw:9,d} {tg:7,d} {ts:7,d} {td+tw+tg+ts:9,d}")
