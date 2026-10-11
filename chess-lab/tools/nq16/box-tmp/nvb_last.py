import sys, json
from collections import Counter

rows = []
for line in open("/extnvme/router_search/features.tsv"):
    p = line.rstrip("\n").split("\t")
    if p[0] == "src":
        continue
    rows.append(p)
print("cache:", len(rows))

# column indices from header:
# 0 src 1 subtag 2 stm 3 men 4 npp 5 q 6 rooks 7 respair 8 oppb 9 lock
# 10 tension 11 pd 12 matd 13 pawndiff 14 cw 15 cb 16 open_t 17 semi_w
# 18 semi_b 19 shield_w 20 shield_b 21 dev_w 22 dev_b 23 bcd 24 ply 25 fen

PAIR = {("B", "N"): "nvb", ("N", "R"): "mvr", ("B", "R"): "mvr",
        ("NN", "R"): "rv2m", ("BN", "R"): "rv2m", ("BB", "R"): "rv2m",
        ("Q", "RR"): "qvmat", ("Q", "RN"): "qvmat", ("Q", "RB"): "qvmat",
        ("BN", "NN"): "n2v2", ("BB", "NN"): "n2v2"}

def route(p, nvb_last):
    men = int(p[3]); npp = int(p[4]); q = int(p[6 - 1])
    respair = p[7]
    pair_key = tuple(sorted(respair.split("|"))) if respair and respair != "-" else None
    if men <= 5:
        return "tb"
    if pair_key is not None and pair_key in PAIR:
        e = PAIR[pair_key]
        if e != "nvb":
            return e
        if not nvb_last:
            return "nvb"
        # nvb deferred: fall through to dv-family / phases
    if pair_key is not None and pair_key != ("", ""):
        # unlisted asymmetric residue -> pdown family (residual folded)
        return "pd_family"
    if npp <= 8 and q <= 2:
        return "dv_family"
    if nvb_last and pair_key == ("B", "") or nvb_last and pair_key == ("", "B") or nvb_last and pair_key == ("N", "B"):
        return "nvb"          # middlegame/late nvb lands here
    if p[23] == "1":
        return "mg_cell"
    return "opening_cell"

for nvb_last in (False, True):
    c = Counter()
    g = Counter()
    for p in rows:
        e = route(p, nvb_last)
        c[e] += 1
        if p[0] == "game":
            g[e] += 1
    n = len(rows)
    gn = sum(g.values())
    print(f"nvb_last={nvb_last}")
    for k in ("nvb", "dv_family", "pd_family", "mg_cell", "opening_cell", "mvr"):
        print(f"  {k:>12}: pooled {100*c.get(k,0)/n:5.2f}%   game {100*g.get(k,0)/gn:5.2f}%")
