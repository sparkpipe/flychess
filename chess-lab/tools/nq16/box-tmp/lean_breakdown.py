"""OPERATOR LEAN — measured: even-openings by locks x tension;
middlegame by castle-state x weak/solid king (composite).
"""
import chess
from collections import Counter

rows = []
for line in open("/extnvme/router_search/features.tsv"):
    p = line.rstrip("\n").split("\t")
    if p[0] != "src":
        rows.append(p)
# cols: 2 stm 3 men 4 npp 5 q 7 respair 10 tension 13 pawndiff
# 19 shield_w 20 shield_b 21 dev_w 22 dev_b 25 fen

def classify(p):
    respair = p[7]
    key = tuple(sorted(respair.split("|"))) if respair not in ("-", "|") else None
    if key is not None and key != ("", ""):
        return None                     # residue layers
    if int(p[3]) <= 5:
        return None
    if int(p[4]) <= 8 and int(p[5]) <= 2:
        return None                     # dv family
    stm = p[2]
    mover_up = (int(p[13]) > 0) if stm == "w" else (int(p[13]) < 0)
    developed = int(p[21]) >= 2 and int(p[22]) >= 2
    if not developed:
        if int(p[13]) != 0:
            return "op_gambiteer" if not mover_up else "op_acceptor"
        lock = int(p[9]) if p[9].isdigit() else 0
        lk = "l0" if lock == 0 else "l1" if lock == 1 else "l2+"
        return f"op_even_{lk}"
    # middlegame: castle 5-way x weak/solid
    b = chess.Board(p[25])
    wk, bk = b.king(chess.WHITE), b.king(chess.BLACK)
    def cast(ksq, white):
        if ksq is None:
            return None
        r0 = 0 if white else 7
        f, r = chess.square_file(ksq), chess.square_rank(ksq)
        return "cK" if (r == r0 and f == 6) else "cQ" if (r == r0 and f == 2) else None
    wc, bc = cast(wk, True), cast(bk, False)
    def unsafe(ksq, white):
        if ksq is None:
            return True
        kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
        sh = 0
        for df in (-1, 0, 1):
            for dr in ((1, 2) if white else (-1, -2)):
                s = chess.square(kf + df, kr + dr) if 0 <= kf + df < 8 and 0 <= kr + dr < 8 else None
                if s is not None:
                    q = b.piece_at(s)
                    if q and q.piece_type == chess.PAWN and q.color == white:
                        sh += 1
        if sh <= 1:
            return True
        if sh <= 2:
            zone = [chess.square(kf + df, kr + dr)
                    for df in (-2, -1, 0, 1, 2) for dr in (-2, -1, 0, 1, 2)
                    if 0 <= kf + df < 8 and 0 <= kr + dr < 8]
            hits = 0
            for s in zone:
                for a in b.attackers(not white, s):
                    q = b.piece_at(a)
                    if q and q.piece_type != chess.PAWN:
                        hits += 1
                        break
            return hits >= 2
        return False
    me, other = (wc, bc) if stm == "w" else (bc, wc)
    myk, ok = (wk, bk) if stm == "w" else (bk, wk)
    myw = unsafe(myk, stm == "w")
    otw = unsafe(ok, stm != "w")
    safety = "weak" if myw else ("solid" if not otw else "solid")   # mover-first
    if me is None and other is None:
        cst = "uncastled"
    elif me is not None and other is not None:
        cst = "both_same" if me == other else "opposite"
    elif me is not None:
        cst = "my_castled"
    else:
        cst = "other_castled"
    # both-weak variant worth showing
    if myw or cst == "opposite":
        return "mg_unsafe_king"
    return f"mg_safe_{cst}"

c = Counter()
n = 0
for p in rows:
    e = classify(p)
    if e:
        c[e] += 1
        n += 1

total_game = sum(1 for p in rows if p[0] == "game")
cg = Counter()
for p in rows:
    if p[0] == "game":
        e = classify(p)
        if e:
            cg[e] += 1

ORDER = ["op_gambiteer", "op_acceptor",
         "op_even_l0", "op_even_l1", "op_even_l2+",
         "mg_unsafe_king",
         "mg_safe_uncastled", "mg_safe_both_same",
         "mg_safe_my_castled", "mg_safe_other_castled"]
print(f"phase-domain positions in cache: {n:,} (game rows: {total_game:,})")
print(f"{'cell':<24} {'cache%':>7} {'game%':>7}")
for k in ORDER:
    v = c.get(k, 0)
    g = cg.get(k, 0)
    print(f"{k:<24} {100*v/n:>6.2f}% {100*g/total_game:>6.2f}%")
