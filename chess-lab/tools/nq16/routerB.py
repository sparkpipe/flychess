"""routerB — revised taxonomy (operator rulings 2026-10-05), priority order:

 1 tb             total pieces <= 5
 2 dv             sparse endgames: <=8 non-pawn pieces, <=2 queens (merged family)
 3 mvr            R v one minor (the exchange)
 4 rv2m           R v exactly two minors
 5 qvmat          Q v two or three pieces (R/N/B)
 6 n2v2           2N v (2B | N+B)
 7 nvb            N v B
 8 piece_down     any other material imbalance (one side down piece(s))
 9 oppb           equal material, opposite bishop-color majorities
10 op_pawnimb     equal pieces, pawn count differs, fullmove < 15 (gambiteer+acceptor)
11-13 op_even_l0/l1/l2+  even, fullmove < 15, some side undeveloped, by center locks
14 mg_unsafe      opposite castling or own king shield <= 1
15 mg_safe_both_same / my_castled / other_castled / uncastled

Approximations for table routing (C++ port will be exact): king shield from
pawn placement; zone-attackers not modeled.
"""
import sys
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
from router12 import _parse


def routeB(fen):
    (cw, cb, wp, bp, pieces, wk, bk, dw, db_, resw, resb, lock, stm, rights) = _parse(fen)
    men = len(pieces) + len(wp) + len(bp)
    if men <= 5:
        return "tb"

    npp = sum(cw.values()) + sum(cb.values())
    q = cw["Q"] + cb["Q"]
    dN = cw["N"] - cb["N"]
    dB = cw["B"] - cb["B"]
    dR = cw["R"] - cb["R"]
    dQ = cw["Q"] - cb["Q"]
    asym = dN or dB or dR or dQ

    if asym:
        def pat(a, b, c, d):
            return (dN, dB, dR, dQ) == (a, b, c, d) or (dN, dB, dR, dQ) == (-a, -b, -c, -d)
        # mvr: R v one minor
        if pat(1, 0, -1, 0) or pat(0, 1, -1, 0):
            return "mvr"
        # rv2m: R v exactly two minors
        if pat(2, 0, -1, 0) or pat(0, 2, -1, 0) or pat(1, 1, -1, 0):
            return "rv2m"
        # qvmat: Q v two or three non-queen pieces
        for (a, b, c) in [(2,0,0),(0,2,0),(0,0,2),(1,1,0),(1,0,1),(0,1,1),
                          (3,0,0),(0,3,0),(0,0,3),(2,1,0),(2,0,1),(1,2,0),
                          (0,2,1),(1,0,2),(0,1,2),(1,1,1)]:
            if pat(a, b, c, -1):
                return "qvmat"
        # n2v2 merged into nvb (operator ruling)
        if pat(2, -2, 0, 0):
            return "nvb"
        # nvb: N v B
        if pat(1, -1, 0, 0):
            return "nvb"
        return "piece_down"

    parts = fen.split()
    try:
        fm = int(parts[5])
    except Exception:
        fm = 1

    # oppb
    wb_sq = [(f, r) for (f, r), ch in pieces.items() if ch == "B"]
    bb_sq = [(f, r) for (f, r), ch in pieces.items() if ch == "b"]
    if wb_sq and bb_sq:
        wod = sum(1 for f, r in wb_sq if (f + r) % 2 == 1)
        bod = sum(1 for f, r in bb_sq if (f + r) % 2 == 1)
        if (2 * wod > len(wb_sq)) != (2 * bod > len(bb_sq)):
            return "oppb"

    if npp <= 8 and q <= 2:
        # dv split: with queens / with rooks no queens / the rest
        if q:
            return "dv_Q"
        if cw["R"] + cb["R"]:
            return "dv_R"
        return "dv_rest"

    if len(wp) != len(bp) and fm < 15:
        return "op_pawnimb"      # openings: unbalanced pawns -> pawnimb (no undeveloped gate)

    if fm < 15:
        if len(wp) != len(bp):
            return "op_pawnimb"
        return "op_even_l2p" if lock >= 2 else ("op_even_l1" if lock == 1 else "op_even_l0")

    # middlegame: king shield from pawns (files +-1, two ranks ahead)
    def shield(k, pawns, white):
        kf, kr = k
        n = 0
        for df in (-1, 0, 1):
            for dr in (1, 2):
                f = kf + df - 1
                r = kr + (dr if white else -dr)
                if 1 <= f <= 8 and 1 <= r <= 8 and (f, r) in pawns:
                    n += 1
        return n
    sw = shield(wk, wp, True)
    sb = shield(bk, bp, False)
    # castled: king on g/c file at home rank with that side's right gone
    def castled(k):
        f, r = k
        return (r in (1, 8)) and (f in (6, 2))  # g/c files, 0-based
    cw_ = castled(wk)
    cb_ = castled(bk)
    if cw_ and cb_:
        if abs(wk[0] - bk[0]) >= 3:  # 0-based: g vs c = 4
            return "mg_unsafe"
        if min(sw, sb) <= 1:
            return "mg_unsafe"
        return "mg_safe"
    if min(sw, sb) <= 1 and not (cw_ or cb_):
        pass
    if cw_ or cb_:
        return "mg_safe"
    if min(sw, sb) <= 1:
        return "mg_unsafe"
    return "mg_safe"
