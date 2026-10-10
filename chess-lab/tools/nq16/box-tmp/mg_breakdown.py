"""MIDDLEGAME BREAKDOWN — castle-state 5-way (mover-relative) x king safety.

Middlegame gate: development (both sides >= 2 minors off home squares).
King safety (better def):
  shield = pawns on the 3 files around the castled king (f,g,h for cK etc.),
           any rank; for uncastled kings: pawns on d,e,f within 2 ranks.
  zone_attackers = enemy non-pawn pieces attacking any square within
                   Chebyshev distance 2 of own king.
Outputs: 5-way castle distribution, king-safety distributions, cross table.
"""
import sys
from collections import Counter

rows = []
for line in open("/extnvme/router_search/features.tsv"):
    p = line.rstrip("\n").split("\t")
    if p[0] != "src":
        rows.append(p)
# cols: 0 src 2 stm 3 men 4 npp 5 q 6 rooks 7 respair 8 oppb 9 lock 10 tension
# 13 pawndiff 16 open_t 17 semi_w 18 semi_b 19 shield_w 20 shield_b
# 21 dev_w 22 dev_b 23 bcd 24 ply 25 fen
import chess

PAIR = {("B", "N"): "nvb", ("N", "R"): "mvr", ("B", "R"): "mvr",
        ("NN", "R"): "rv2m", ("BN", "R"): "rv2m", ("BB", "R"): "rv2m",
        ("Q", "RR"): "qvmat", ("Q", "RN"): "qvmat", ("Q", "RB"): "qvmat",
        ("BN", "NN"): "n2v2", ("BB", "NN"): "n2v2"}

def mg_state(p):
    fen = p[25]
    b = chess.Board(fen)
    stm = p[2]
    wk = b.king(chess.WHITE)
    bk = b.king(chess.BLACK)
    def cast(ksq, white):
        r0 = 0 if white else 7
        if ksq is None:
            return None
        f, r = chess.square_file(ksq), chess.square_rank(ksq)
        if r != r0:
            return None
        if f == 6:
            return "cK"
        if f == 2:
            return "cQ"
        return None
    wc, bc = cast(wk, True), cast(bk, False)
    me, other = (wc, bc) if stm == "w" else (bc, wc)
    if me is None and other is None:
        st = "mg_uncastled"
    elif me is not None and other is not None:
        st = "mg_both_same" if me == other else "mg_opposite"
    elif me is not None:
        st = "mg_my_castled"
    else:
        st = "mg_other_castled"
    return st, wc, bc, b, wk, bk

castle = Counter()
shield = Counter()
attackers = Counter()
cross = Counter()
n_mg = 0
for p in rows:
    # middlegame gate: developed both sides, not in earlier router layers
    respair = p[7]
    key = tuple(sorted(respair.split("|"))) if respair not in ("-", "|") else None
    if key is not None and key != ("", ""):
        continue                      # residue layers own these
    if int(p[4]) <= 8 and int(p[5]) <= 2:
        continue                      # dv family
    if int(p[21]) < 2 or int(p[22]) < 2:
        continue                      # not developed: opening phase
    n_mg += 1
    st, wc, bc, b, wk, bk = mg_state(p)
    castle[st] += 1
    # king safety: shield pawns per side + zone attackers per side
    for white, ksq, tag in ((True, wk, "w"), (False, bk, "b")):
        if ksq is None:
            continue
        kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
        sh = 0
        for df in (-1, 0, 1):
            for dr in ((1, 2) if white else (-1, -2)):
                s = chess.square(kf + df, kr + dr) if 0 <= kf + df < 8 and 0 <= kr + dr < 8 else None
                if s is not None:
                    q = b.piece_at(s)
                    if q and q.piece_type == chess.PAWN and q.color == white:
                        sh += 1
        shield[(tag, min(sh, 3))] += 1
        zone = [chess.square(kf + df, kr + dr)
                for df in (-2, -1, 0, 1, 2) for dr in (-2, -1, 0, 1, 2)
                if 0 <= kf + df < 8 and 0 <= kr + dr < 8]
        atk = 0
        for s in zone:
            for a in b.attackers(not white, s):
                q = b.piece_at(a)
                if q and q.piece_type != chess.PAWN:
                    atk += 1
                    break
        attackers[(tag, min(atk, 4))] += 1
    stm_tag = "w" if p[2] == "w" else "b"
    sh_mine = 0
    # quick: mover shield from cross of shield dict — recompute below instead
    cross[st] += 0

n2 = n_mg * 2
print(f"middlegame positions (developed, no residue, non-dv): {n_mg:,}")
print("\ncastle-state 5-way (mover-relative):")
for k, v in castle.most_common():
    print(f"  {k:>16} {100*v/n_mg:5.2f}%  {v:>8,}")
print("\nshield pawns distribution (per king, both sides pooled):")
for t in ("w", "b"):
    line = f"  {t}: "
    for s in range(4):
        line += f"{s}={100*shield.get((t,s),0)/n_mg:5.1f}% "
    print(line)
print("\nzone attackers (non-pawn pieces hitting king zone, capped 4):")
for t in ("w", "b"):
    line = f"  {t}: "
    for a in range(5):
        line += f"{a}={100*attackers.get((t,a),0)/n_mg:5.1f}% "
    print(line)
