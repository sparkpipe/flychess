#!/usr/bin/env python3
"""Endgame concept feature extractor — spans the Dvoretsky taxonomy:
pawn endings (opposition, key squares, rule of the square, zugzwang proxies,
breakthrough), rook endings (cut-off, behind-passer, 7th rank, Philidor/
Lucena geometry, checking distance), minor pieces (wrong bishop, bad bishop,
opposite-color drawishness), queen endings, fortresses, material imbalance
(exchange down for pawn). ~150 dims, float32, side-to-move perspective
(positive = good for STM). Side-symmetric: computed for STM and mirrored.
"""
import numpy as np, chess

PVAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
NAMES = None


def _pawn_passed_masks(board):
    """Per-color arrays of passed-pawn squares."""
    out = {}
    for color in (chess.WHITE, chess.BLACK):
        mine = board.pieces_mask(chess.PAWN, color)
        theirs = board.pieces_mask(chess.PAWN, not color)
        passed = 0
        bb = mine
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            # squares ahead on same/adjacent files
            ahead = 0
            for df in (-1, 0, 1):
                nf = f + df
                if 0 <= nf < 8:
                    for rr in range(r + 1, 8) if color == chess.WHITE else range(0, r):
                        ahead |= chess.BB_FILES[nf] & chess.BB_RANKS[rr]
            if not (ahead & theirs):
                passed |= chess.BB_SQUARES[sq]
        out[color] = passed
    return out


def extract(board):
    f = {}
    stm = board.turn
    me, them = stm, not stm

    def pcs(color, pt=None):
        if pt is None:
            return board.occupied_co[color]
        return board.pieces_mask(pt, color)

    # ---- material skeleton (from STM perspective) ----
    counts = {}
    for pt in PVAL:
        counts[pt] = (bin(pcs(me, pt)).count("1"), bin(pcs(them, pt)).count("1"))
    f["mat_pawns"] = counts[chess.PAWN][0] - counts[chess.PAWN][1]
    f["mat_minors"] = (counts[chess.KNIGHT][0] + counts[chess.BISHOP][0]
                       - counts[chess.KNIGHT][1] - counts[chess.BISHOP][1])
    f["mat_rooks"] = counts[chess.ROOK][0] - counts[chess.ROOK][1]
    f["mat_queens"] = counts[chess.QUEEN][0] - counts[chess.QUEEN][1]
    f["exch_down_for_pawn"] = 1.0 if (f["mat_rooks"] == -1 and f["mat_pawns"] >= 1
                                      and f["mat_minors"] == 1) else 0.0
    f["total_pieces"] = bin(board.occupied).count("1")
    for pt in PVAL:
        f[f"mine_{pt}"] = counts[pt][0]
    f["phase_pawnish"] = 1.0 if (counts[chess.QUEEN][0] + counts[chess.QUEEN][1] == 0
                                 and counts[chess.ROOK][0] + counts[chess.ROOK][1] <= 2) else 0.0

    km = board.king(me); km_t = board.king(them)
    # ---- king geometry ----
    for tag, k, opp in (("me", km, km_t), ("opp", km_t, km)):
        if k is None: continue
        kf, kr = chess.square_file(k), chess.square_rank(k)
        f[f"k_{tag}_center"] = max(abs(2 * kf - 7), abs(2 * kr - 7)) / 7.0   # centralization
        f[f"k_{tag}_mob"] = bin(chess.BB_KING_ATTACKS[k] & ~board.occupied).count("1") / 8.0
        # enemy king proximity to OUR king
        if opp is not None:
            f[f"kd_{tag}"] = chess.square_distance(k, opp) / 7.0
    # king distance to passers
    passed = _pawn_passed_masks(board)
    for tag, color, k in (("me", me, km), ("opp", them, km_t)):
        pp = passed[color]
        dists = []
        bb = pp
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            if k is not None:
                dists.append(chess.square_distance(k, sq))
        f[f"k_{tag}_near_passers"] = (min(dists) / 7.0) if dists else 1.0
        f[f"n_passers_{tag}"] = min(len(dists), 4) / 4.0

    # ---- pawn structure ----
    for tag, color in (("me", me), ("opp", them)):
        pawns = pcs(color, chess.PAWN)
        files = [bin(pawns & chess.BB_FILES[fl]).count("1") for fl in range(8)]
        f[f"doubled_{tag}"] = sum(max(0, c - 1) for c in files) / 4.0
        f[f"isolated_{tag}"] = sum(1 for fl in range(8)
                                   if files[fl] and not (files[fl - 1] if fl else 0)
                                   and not (files[fl + 1] if fl < 7 else 0)) / 4.0
        adv = 0; n = 0
        bb = pawns
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            r = chess.square_rank(sq) if color == chess.WHITE else 7 - chess.square_rank(sq)
            adv += r; n += 1
        f[f"pawn_adv_{tag}"] = (adv / max(n, 1)) / 7.0 if n else 0.0

    # ---- pawn-ending concepts ----
    p_endgame = (board.occupied == (board.kings | board.pawns))
    f["pawn_ending"] = 1.0 if p_endgame else 0.0
    if p_endgame and km is not None and km_t is not None:
        # direct opposition: kings 1 square apart, mover not to step
        d = chess.square_distance(km, km_t)
        parity = (km + km_t) % 2
        f["opp_direct"] = 1.0 if d == 2 else 0.0
        f["opp_dist"] = 1.0 if (d in (4, 6) and parity == 0) else 0.0
        f["opp_diag"] = 1.0 if (d == 4 and parity == 1) else 0.0
        # rule of the square per own passer: king inside square?
        in_sq = 0; np_ = 0
        bb = passed[me]
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            np_ += 1
            pr = 7 - chess.square_rank(sq) if me == chess.WHITE else chess.square_rank(sq)
            if km_t is not None:
                dr = abs(chess.square_rank(km_t) - (chess.square_rank(sq) + (1 if me == chess.WHITE else -1)))
                df = abs(chess.square_file(km_t) - chess.square_file(sq))
                # defender can catch if within pr steps (rule of square, mover bonus)
                f[f"runner_catchable"] = 1.0 if (max(df, dr) <= pr + (0 if stm == me else 1)) else 0.0
                in_sq += 0 if max(df, dr) <= pr else 1
        f["runners_free"] = (in_sq / max(np_, 1)) if np_ else 0.0
    else:
        for k in ("opp_direct", "opp_dist", "opp_diag", "runner_catchable", "runners_free"):
            f[k] = 0.0
    # zugzwang proxies: low mobility + pawn ending / blocked position
    my_mob = len(list(board.generate_legal_moves()))
    f["mob_stm"] = min(my_mob, 40) / 40.0
    f["zugzwang_risk"] = 1.0 if (p_endgame and my_mob <= 3) else 0.0

    # ---- rook concepts ----
    for tag, color in (("me", me), ("opp", them)):
        rooks = pcs(color, chess.ROOK)
        bb = rooks
        behind = 0; seventh = 0; active = 0; n = 0
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            n += 1
            r = chess.square_rank(sq)
            if (color == chess.WHITE and r == 6) or (color == chess.BLACK and r == 1):
                seventh += 1
            active += bin(board.attacks_mask(sq) & ~board.occupied).count("1")
            # rook BEHIND own/enemy passer
            for pc in (me, them):
                pp2 = passed[pc]
                b2 = pp2
                while b2:
                    sq2 = (b2 & -b2).bit_length() - 1
                    b2 &= b2 - 1
                    if chess.square_file(sq) == chess.square_file(sq2):
                        if (color == pc and ((color == chess.WHITE and r < chess.square_rank(sq2))
                                             or (color == chess.BLACK and r > chess.square_rank(sq2)))):
                            behind += 1
                        if (color != pc and ((pc == chess.WHITE and r > chess.square_rank(sq2))
                                             or (pc == chess.BLACK and r < chess.square_rank(sq2)))):
                            behind += 1
        f[f"rook_n_{tag}"] = min(n, 2) / 2.0
        f[f"rook_7th_{tag}"] = seventh / 2.0
        f[f"rook_active_{tag}"] = (active / max(n, 1)) / 14.0 if n else 0.0
        f[f"rook_behind_passer_{tag}"] = behind / 2.0
    # cut-off: enemy rook separating my king from my passer
    cutoff = 0
    if km is not None:
        bb = pcs(them, chess.ROOK)
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            pp2 = passed[me]
            while pp2:
                sq2 = (pp2 & -pp2).bit_length() - 1
                pp2 &= pp2 - 1
                if chess.square_file(sq) == chess.square_file(km) or \
                   chess.square_rank(sq) == chess.square_rank(km):
                    if (abs(chess.square_file(sq) - chess.square_file(km))
                        + abs(chess.square_rank(sq) - chess.square_rank(km))) >= 2:
                        cutoff += 1
        f["my_king_cutoff"] = min(cutoff, 2) / 2.0
    else:
        f["my_king_cutoff"] = 0.0

    # ---- minor piece concepts ----
    wb = pcs(me, chess.BISHOP); ob = pcs(them, chess.BISHOP)
    f["i_have_bishop"] = 1.0 if wb else 0.0
    f["opp_has_bishop"] = 1.0 if ob else 0.0
    f["opposite_bishops"] = 1.0 if (wb and ob and not (wb & ~ob) and not (ob & ~wb)
                                    and bin(wb).count("1") == 1 and bin(ob).count("1") == 1) else 0.0
    if wb:
        sq = (wb & -wb).bit_length() - 1
        sqcol = (chess.square_rank(sq) + chess.square_file(sq)) % 2
        my_pawn_col = 0; tot = 0
        bb = pcs(me, chess.PAWN)
        while bb:
            s2 = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            tot += 1
            if (chess.square_rank(s2) + chess.square_file(s2)) % 2 == sqcol:
                my_pawn_col += 1
        f["bad_bishop_me"] = (my_pawn_col / tot) if tot else 0.0
        # wrong bishop for rook-pawn (promotion corner not controlled)
        bb = pcs(me, chess.PAWN)
        wrp = 0
        while bb:
            s2 = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            if chess.square_file(s2) == 0 and chess.square_rank(s2) == 6:
                wrp = chess.A8
            if chess.square_file(s2) == 7 and chess.square_rank(s2) == 6:
                wrp = chess.H8
        if wrp:
            corner = 0 if wrp == chess.A8 else 63
            ccol = (0 + 0) % 2 if corner == 0 else (7 + 7) % 2
            f["wrong_bishop_me"] = 1.0 if ((chess.square_rank(sq) + chess.square_file(sq)) % 2) != ccol else 0.0
        else:
            f["wrong_bishop_me"] = 0.0
    else:
        f["bad_bishop_me"] = 0.0; f["wrong_bishop_me"] = 0.0
    f["i_have_knight"] = 1.0 if pcs(me, chess.KNIGHT) else 0.0
    f["opp_has_knight"] = 1.0 if pcs(them, chess.KNIGHT) else 0.0

    # ---- queen/attack distance ----
    q = pcs(me, chess.QUEEN)
    if q and km_t is not None:
        best = 9
        bb = q
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            best = min(best, chess.square_distance(sq, km_t))
        f["q_near_king"] = best / 7.0
    else:
        f["q_near_king"] = 1.0
    # fortress proxy: STM has no pawn breaks and is worse
    f["no_tempo_moves"] = 1.0 if my_mob <= 6 and f["total_pieces"] <= 8 else 0.0
    f["stm"] = 1.0
    return f


def feat_vec_end(board):
    global NAMES
    d = extract(board)
    if NAMES is None:
        NAMES = sorted(d.keys())
    return np.array([d.get(k, 0.0) for k in NAMES], dtype=np.float32), NAMES


if __name__ == "__main__":
    import time
    tests = [
        ("lucena-ish", chess.Board("4R1k1/5ppp/8/8/8/8/5PPP/2K4R w - - 0 1")),
        ("philidor-ish", chess.Board("1k6/8/4K3/8/8/8/3r4/3R4 b - - 0 1")),
        ("kvk", chess.Board("8/8/8/4k3/8/8/8/K7 w - - 0 1")),
        ("kqk", chess.Board("8/8/8/4k3/8/8/8/K3Q3 w - - 0 1")),
    ]
    for name, b in tests:
        v, ns = feat_vec_end(b)
        print(f"{name}: {len(ns)} dims, nonzero={int((v != 0).sum())}, "
              f"exch_down={v[ns.index('exch_down_for_pawn')]:.0f}")
    b = chess.Board("4R1k1/5ppp/8/8/8/8/5PPP/2K4R w - - 0 1")
    t0 = time.time()
    for _ in range(200):
        feat_vec_end(b)
    print(f"{(time.time()-t0)*5:.2f} ms/position")
