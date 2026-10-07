"""UNIFIED 23-EXPERT ROUTE — single pass, exact totals, router order."""
import chess
from collections import Counter

PAIR = {("N", "R"): "mvr", ("B", "R"): "mvr",
        ("NN", "R"): "rv2m", ("BN", "R"): "rv2m", ("BB", "R"): "rv2m",
        ("Q", "RR"): "qvmat", ("Q", "RN"): "qvmat", ("Q", "RB"): "qvmat",
        ("BN", "NN"): "n2v2", ("BB", "NN"): "n2v2"}

rows = []
for line in open("/extnvme/router_search/features.tsv"):
    p = line.rstrip("\n").split("\t")
    if p[0] != "src":
        rows.append(p)

def route(p):
    stm = p[2]
    men, npp, q, rooks = int(p[3]), int(p[4]), int(p[5]), int(p[6])
    respair = p[7]
    key = tuple(sorted(respair.split("|"))) if respair not in ("-", "|") else None
    asym = key is not None and key != ("", "")
    mover_up = (int(p[13]) > 0) if stm == "w" else (int(p[13]) < 0)
    if men <= 5:
        return "tb"
    if asym and key in PAIR:
        return PAIR[key]
    if asym and key != ("B", "N"):
        return "pd_down" if not mover_up else "pd_up"
    if p[8] == "1":
        return "oppb"
    if npp <= 8 and q <= 2:
        if rooks and q: return "dv_QRend"
        if rooks: return "dv_rend"
        if q: return "dv_qend"
        return "dv_core"
    if asym and key == ("B", "N"):
        return "nvb"
    developed = int(p[21]) >= 2 and int(p[22]) >= 2
    if not developed:
        if int(p[13]) != 0:
            return "op_gambiteer" if not mover_up else "op_acceptor"
        lock = int(p[9]) if p[9].isdigit() else 0
        return ["op_even_l0", "op_even_l1", "op_even_l2+"][min(lock, 2)]
    b = chess.Board(p[25])
    wk, bk = b.king(chess.WHITE), b.king(chess.BLACK)
    def cast(ksq, white):
        if ksq is None: return None
        r0 = 0 if white else 7
        f, r = chess.square_file(ksq), chess.square_rank(ksq)
        return "cK" if (r == r0 and f == 6) else "cQ" if (r == r0 and f == 2) else None
    def unsafe(ksq, white):
        if ksq is None: return True
        kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
        sh = 0
        for df in (-1, 0, 1):
            for dr in ((1, 2) if white else (-1, -2)):
                s = chess.square(kf + df, kr + dr) if 0 <= kf + df < 8 and 0 <= kr + dr < 8 else None
                if s is not None:
                    pc = b.piece_at(s)
                    if pc and pc.piece_type == chess.PAWN and pc.color == white:
                        sh += 1
        if sh <= 1: return True
        if sh <= 2:
            zone = [chess.square(kf + df, kr + dr)
                    for df in (-2, -1, 0, 1, 2) for dr in (-2, -1, 0, 1, 2)
                    if 0 <= kf + df < 8 and 0 <= kr + dr < 8]
            hits = 0
            for s in zone:
                for a in b.attackers(not white, s):
                    pc = b.piece_at(a)
                    if pc and pc.piece_type != chess.PAWN:
                        hits += 1
                        break
            return hits >= 2
        return False
    wc, bc = cast(wk, True), cast(bk, False)
    me, other = (wc, bc) if stm == "w" else (bc, wc)
    myk = wk if stm == "w" else bk
    if unsafe(myk, stm == "w"):
        return "mg_unsafe_king"
    if me is not None and other is not None and me != other:
        return "mg_unsafe_king"           # opposite castling = both unsafe
    if me is None and other is None:
        return "mg_safe_uncastled"
    if me is not None and other is not None:
        return "mg_safe_both_same"
    if me is not None:
        return "mg_safe_my_castled"
    return "mg_safe_other_castled"

ORDER = ["tb", "mvr", "rv2m", "qvmat", "n2v2", "pd_down", "pd_up", "oppb",
         "dv_rend", "dv_QRend", "dv_qend", "dv_core", "nvb",
         "op_gambiteer", "op_acceptor", "op_even_l0", "op_even_l1", "op_even_l2+",
         "mg_unsafe_king", "mg_safe_both_same", "mg_safe_my_castled",
         "mg_safe_uncastled", "mg_safe_other_castled"]

g = Counter()
n = 0
for p in rows:
    if p[0] != "game":
        continue
    g[route(p)] += 1
    n += 1
tot = sum(g[k] for k in ORDER)
print(f"game rows routed: {n:,}  ordered-sum: {tot:,}  "
      f"(unclassified/other: {n - tot:,})")
for i, k in enumerate(ORDER, 1):
    print(f"{i:>2} {k:<22} {100*g[k]/n:5.2f}%  {g[k]:>8,}")
print(f"   {'TOTAL':<22} {100*tot/n:5.2f}%")
