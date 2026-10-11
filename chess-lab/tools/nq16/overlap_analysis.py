"""Overlap between main (train16) and anti (anti16) training sets.
Exact-FEN intersection via hash sets; per-expert breakdown."""
import os, hashlib

def fen_hashes(bindir):
    # decode FEN from 40-byte records is costly; instead re-derive from sources:
    pass

# Cheaper + exact: re-read the segment TSVs and split by sig, intersect by fen.
SEG = "/extnvme/segments"
main_fens = set()
anti_fens = set()
import sys
sys.path.insert(0, "/tmp")
from routerB import routeB
from collections import Counter
main_e, anti_e = Counter(), Counter()
overlap_e = Counter()
for fn in sorted(os.listdir(SEG)):
    if not (fn.startswith("pos_") and (fn.endswith(".clean.tsv") or fn.endswith(".draws.tsv"))):
        continue
    for line in open(f"{SEG}/{fn}", errors="ignore"):
        p = line.rstrip("\n").split("|")
        if len(p) < 8 or p[3] not in ("pos", "anti"):
            continue
        fen = p[0]
        if p[3] == "pos":
            main_fens.add(fen); main_e[routeB(fen)] += 1
        else:
            anti_fens.add(fen); anti_e[routeB(fen)] += 1
inter = main_fens & anti_fens
for fen in inter:
    overlap_e[routeB(fen)] += 1
tot_m, tot_a = len(main_fens), len(anti_fens)
print(f"main unique positions: {tot_m:,}")
print(f"anti unique positions: {tot_a:,}")
print(f"EXACT-FEN OVERLAP: {len(inter):,} ({100*len(inter)/max(1,tot_a):.1f}% of anti, {100*len(inter)/max(1,tot_m):.1f}% of main)")
print(f"{'expert':18s} {'main':>10s} {'anti':>10s} {'overlap':>9s} {'ov%_anti':>8s}")
for e in sorted(set(main_e) | set(anti_e)):
    m, a, o = main_e.get(e,0), anti_e.get(e,0), overlap_e.get(e,0)
    print(f"{e:18s} {m:10,d} {a:10,d} {o:9,d} {100*o/max(1,a):7.1f}%")
