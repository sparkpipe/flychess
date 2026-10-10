"""SEGMENT STEEPNESS BY OUTCOME — per-segment, not per-position.

For each pos-sig segment: plies spent inside the band = steepness
(fixed wp range per band, so fewer plies = steeper climb).
Group by band x eventual game result. Hypothesis (operator): winner-side
segments are steeper.
"""
import os
from collections import defaultdict

SEG = "/mnt/cold-raid6/chess-audit/wp_fit/segments"
UP = ["30to45", "45to60", "55to70", "70to100"]

# seg_index columns: seg_id gid white black result welo belo date event side band sig n_plys ply_lo ply_hi
acc = defaultdict(lambda: [0, 0, 0])   # (band, result) -> [n_segments, sum_plies, sum_plies_sq]
for fname, res_col in (("seg_index.tsv", 4), ("seg_index.draws.tsv", 4)):
    path = f"{SEG}/{fname}"
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 15 or p[11] != "pos" or p[10] not in UP:
                continue
            band, res = p[10], p[4]
            nplies = int(p[12])
            a = acc[(band, res)]
            a[0] += 1
            a[1] += nplies
            a[2] += nplies * nplies

import math
print("pos-slope SEGMENTS: mean plies per segment (steeper = fewer plies)")
print(f"{'band':>8} {'result':>8} {'segs':>9} {'mean plies':>10} {'median~':>8}")
for band in UP:
    for res in ("1-0", "0-1", "1/2-1/2"):
        n, s, sq = acc.get((band, res), [0, 0, 0])
        if n < 100:
            continue
        mean = s / n
        # median approx via mean/variance (lognormal-ish): use mean and sqrt
        sd = math.sqrt(max(sq / n - mean * mean, 0.01))
        med = math.exp(math.log(max(mean, 0.1)) - 0.5 * math.log(1 + (sd / mean) ** 2))
        print(f"{band:>8} {res:>8} {n:>9,} {mean:>10.1f} {med:>8.1f}")
