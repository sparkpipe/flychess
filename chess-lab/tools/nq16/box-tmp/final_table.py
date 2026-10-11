"""FINAL EXPERT TABLE — variant F, nvb-after-dv (ruling), pooled with all sources.

Cache: /extnvme/router_search/features.tsv (stratified sample with src tags).
Scale to populations: game 2,243,187 / degm 44,550 / puzzles 6,100,952 / tb 12,547.
Also print pooled at 50% puzzle cap (puzzle_n capped so puzzles = 50% of pool).
"""
from collections import Counter

game_n, degm_n, puz_n, tb_n = 2_243_187, 44_550, 6_100_952, 12_547

rows = []
for line in open("/extnvme/router_search/features.tsv"):
    p = line.rstrip("\n").split("\t")
    if p[0] == "src":
        continue
    rows.append(p)
# cols: 0 src 2 stm 3 men 4 npp 5 q 6 rooks 7 respair 8 oppb 9 lock 10 tension
#       13 pawndiff 16 open_t 19 shield_w 20 shield_b 23 bcd   (cw cb at 14 15)

PAIR = {("B", "N"): "nvb", ("N", "R"): "mvr", ("B", "R"): "mvr",
        ("NN", "R"): "rv2m", ("BN", "R"): "rv2m", ("BB", "R"): "rv2m",
        ("Q", "RR"): "qvmat", ("Q", "RN"): "qvmat", ("Q", "RB"): "qvmat",
        ("BN", "NN"): "n2v2", ("BB", "NN"): "n2v2"}

def route(p):
    stm, men, npp, q, rooks = p[2], int(p[3]), int(p[4]), int(p[5]), int(p[6])
    respair = p[7]
    parts = respair.split("|")
    key = tuple(sorted(parts)) if respair not in ("-", "|") else None
    asym = key is not None and key != ("", "")
    mover_up = (int(p[13]) > 0) if stm == "w" else (int(p[13]) < 0)
    if men <= 5:
        return "tb"
    if asym and key in PAIR and PAIR[key] != "nvb":
        return PAIR[key]
    if asym and key != ("B", "N"):
        # piece-value difference (residual equal-value folded into up)
        return "pd_down" if not mover_up else "pd_up"
    if p[8] == "1":
        return "oppb"
    if npp <= 8 and q <= 2:
        if rooks and q:
            return "dv_QRend"
        if rooks:
            return "dv_rend"
        if q:
            return "dv_qend"
        return "dv_core"
    if asym and key == ("B", "N"):
        return "nvb"                      # nvb LAST: only non-ending B-v-N
    if p[23] != "1":
        pd = int(p[13])
        if pd != 0:
            return "op_gambiteer" if not mover_up else "op_acceptor"
        cw, cb = p[14], p[15]
        if cw in ("cK", "cQ") and cb in ("cK", "cQ"):
            return "op_even_both"
        if cw in ("cK", "cQ") or cb in ("cK", "cQ"):
            return "op_even_oneside"
        return "op_even_neither"
    sw, sb = int(p[19]), int(p[20])
    if (cw2 := p[14]) in ("cK", "cQ") and sw <= 1:
        return "mg_weakking"
    if p[15] in ("cK", "cQ") and sb <= 1 and p[14] not in ("cK", "cQ"):
        return "mg_weakking"
    if p[10] == "1":
        return "mg_tension"
    if int(p[16]) >= 1:
        return "mg_openfile"
    if int(p[13]) != 0:
        return "mg_pawn_down" if not mover_up else "mg_pawn_up"
    if int(p[9]) == 0:
        return "mg_lock0"
    return "mg_lock12"

c = {}
for src, pop in (("game", game_n), ("puzzle", puz_n), ("degm", degm_n), ("tb", tb_n)):
    sub = Counter(route(p) for p in rows if p[0] == src)
    tot = sum(sub.values())
    c[src] = {k: v * pop / tot for k, v in sub.items()}

ORDER = ["tb", "mvr", "rv2m", "qvmat", "n2v2", "pd_down", "pd_up", "oppb",
         "dv_rend", "dv_QRend", "dv_qend", "dv_core", "nvb",
         "op_gambiteer", "op_acceptor", "op_even_neither", "op_even_oneside", "op_even_both",
         "mg_weakking", "mg_tension", "mg_openfile", "mg_pawn_down", "mg_pawn_up",
         "mg_lock0", "mg_lock12"]

G = {k: c["game"].get(k, 0) for k in ORDER}
P = {k: sum(c[s].get(k, 0) for s in c) for k in ORDER}
pool_total = sum(P.values())
# 50% cap: puzzles = 50% of pool -> puzzle_budget = (game+degm+tb)
others = pool_total - sum(P[k] for k in ORDER) * 0 + (game_n + degm_n + tb_n)
puz_budget = game_n + degm_n + tb_n
puz_now = sum(c["puzzle"].get(k, 0) for k in ORDER)
scale = min(1.0, puz_budget / puz_now)
C = {k: c["game"].get(k, 0) + c["degm"].get(k, 0) + c["tb"].get(k, 0)
        + c["puzzle"].get(k, 0) * scale for k in ORDER}
cap_total = sum(C.values())

print(f"pool full = {pool_total:,.0f}   pool cap50 = {cap_total:,.0f} (puzzle scale {scale:.3f})")
print(f"{'#':>2} {'expert':<18} {'game':>9} {'DEGM':>7} {'puzz':>9} {'tb-gen':>7} {'cap50':>9} {'cap%':>6} {'game%':>6}")
for i, k in enumerate(ORDER, 1):
    pz = c["puzzle"].get(k, 0) * scale
    dg = c["degm"].get(k, 0)
    tbg = c["tb"].get(k, 0)
    print(f"{i:>2} {k:<18} {G[k]:>9,.0f} {dg:>7,.0f} {pz:>9,.0f} {tbg:>7,.0f} {C[k]:>9,.0f} {100*C[k]/cap_total:>5.2f}% {100*G[k]/game_n:>5.2f}%")
