"""oppb audit: among positions routing to oppb, what are the per-side bishop counts?"""
import sys, os
from collections import Counter
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
from router12 import _parse
sys.path.insert(0, "/tmp")
from routerB import routeB

SEG = "/extnvme/segments"
dist = Counter()
n = 0
oppb_hits = 0
for fn in sorted(os.listdir(SEG)):
    if not (fn.startswith("pos_") and (fn.endswith(".clean.tsv") or fn.endswith(".draws.tsv"))):
        continue
    for line in open(f"{SEG}/{fn}", errors="ignore"):
        p = line.rstrip("\n").split("|")
        if len(p) < 6 or p[3] != "pos":
            continue
        n += 1
        if n % 2 != 0:   # sample half for speed
            continue
        if routeB(p[0]) != "oppb":
            continue
        (cw, cb, *_rest) = _parse(p[0])
        wb = cw["B"]
        bb = cb["B"]
        dist[(wb, bb)] += 1
        oppb_hits += 1
        if oppb_hits >= 200000:
            break
    if oppb_hits >= 200000 or n >= 2000000:
        break
print(f"sampled ~{n:,} positions; oppb-routed: {oppb_hits:,}")
print("per-side bishop counts (whiteB, blackB) among oppb-routed:")
for k, v in dist.most_common(10):
    pct = 100 * v / max(1, oppb_hits)
    print(f"  {k[0]}B v {k[1]}B: {v:,} ({pct:.1f}%)")
