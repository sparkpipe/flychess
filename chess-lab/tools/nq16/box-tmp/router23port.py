p = "/srv/workspace/flychess/src/Stockfish-act/src/phase_moe.h"
t = open(p).read()

head = open("/tmp/phase_moe23.h.txt").read()

# Replace the whole routing section from the enum through end of route fn
start = t.index("/*")
end = t.index("}  // namespace Stockfish")
new_body = head + r'''
#ifndef PHASE_MOE_H_INCLUDED
#define PHASE_MOE_H_INCLUDED

#include <algorithm>

#include "bitboard.h"
#include "position.h"
#include "types.h"

namespace Stockfish {

constexpr int PhaseMoESlots = 23;

enum PhaseMoESlot : int {
    EX_TB = 0, EX_MVR = 1, EX_RV2M = 2, EX_QVMAT = 3, EX_N2V2 = 4,
    EX_PD_DOWN = 5, EX_PD_UP = 6, EX_OPPB = 7,
    EX_DV_REND = 8, EX_DV_QREND = 9, EX_DV_QEND = 10, EX_DV_CORE = 11,
    EX_NVBLATE = 12,
    EX_OP_GAMBiTEER = 13, EX_OP_ACCEPTOR = 14,
    EX_OP_EVEN_L0 = 15, EX_OP_EVEN_L1 = 16, EX_OP_EVEN_L2 = 17,
    EX_MG_UNSAFE = 18, EX_MG_SAME = 19, EX_MG_MY = 20,
    EX_MG_UNCASTLED = 21, EX_MG_OTHER = 22,
};

static_assert(EX_TB == 0 && EX_MG_OTHER == PhaseMoESlots - 1,
              "phase-moe slot ids must be dense 0..PhaseMoESlots-1");

inline int center_locked_files(const Position& pos) {
    const Bitboard wp = pos.pieces(WHITE, PAWN), bp = pos.pieces(BLACK, PAWN);
    int lock = 0;
    for (File f = FILE_A; f <= FILE_H; ++f)
    {
        const Bitboard fw = wp & file_bb(f), fb = bp & file_bb(f);
        if (!fw || !fb)
            continue;
        Rank maxW = RANK_1, minB = RANK_8;
        for (Bitboard b = fw; b;)
        {
            Square s = pop_lsb(b);
            maxW = std::max(maxW, rank_of(s));
        }
        for (Bitboard b = fb; b;)
        {
            Square s = pop_lsb(b);
            minB = std::min(minB, rank_of(s));
        }
        if (minB - maxW == 1 && f >= FILE_C && f <= FILE_F)
            ++lock;
    }
    return lock;
}

struct NppSide { int n, b, r, q; };

inline NppSide npp_side(const Position& pos, Color c) {
    return {pos.count<KNIGHT>(c), pos.count<BISHOP>(c), pos.count<ROOK>(c), pos.count<QUEEN>(c)};
}

inline bool opp_bishops(const Position& pos) {
    if (!pos.count<BISHOP>(WHITE) || !pos.count<BISHOP>(BLACK))
        return false;
    // majority color: count bishops on (file+rank) odd squares
    int wOdd = 0, wN = 0, bOdd = 0, bN = 0;
    for (Bitboard b = pos.pieces(WHITE, BISHOP); b;)
    {
        Square s = pop_lsb(b);
        wOdd += ((int(s) >> 3) + (int(s) & 7)) & 1;
        ++wN;
    }
    for (Bitboard b = pos.pieces(BLACK, BISHOP); b;)
    {
        Square s = pop_lsb(b);
        bOdd += ((int(s) >> 3) + (int(s) & 7)) & 1;
        ++bN;
    }
    bool wm = 2 * wOdd > wN, bm = 2 * bOdd > bN;
    return wm != bm;
}

inline int minors_developed(const Position& pos, Color c) {
    int n = 0;
    for (Bitboard b = pos.pieces(c, KNIGHT) | pos.pieces(c, BISHOP); b;)
    {
        Square s = pop_lsb(b);
        Rank  r = rank_of(s);
        File  f = file_of(s);
        if (c == WHITE && r == RANK_1 && f >= FILE_B && f <= FILE_G)
            continue;
        if (c == BLACK && r == RANK_8 && f >= FILE_B && f <= FILE_G)
            continue;
        ++n;
    }
    return n;
}

// pawn shield around own king (files +-1, 1-2 ranks ahead)
inline int king_shield(const Position& pos, Color c) {
    Square ksq = pos.square<KING>(c);
    int  kf = int(file_of(ksq)), kr = int(rank_of(ksq));
    int  sh = 0;
    for (int df = -1; df <= 1; ++df)
    {
        if (kf + df < 0 || kf + df > 7)
            continue;
        for (int dr = 1; dr <= 2; ++dr)
        {
            int r = kr + (c == WHITE ? dr : -dr);
            if (r < 0 || r > 7)
                continue;
            Square s = Square((r << 3) | (kf + df));
            Piece pc = pos.piece_on(s);
            if (pc != NO_PIECE && type_of(pc) == PAWN && color_of(pc) == c)
                ++sh;
        }
    }
    return sh;
}

// non-pawn attackers hitting any square within chebyshev 2 of own king
inline int king_zone_attackers(const Position& pos, Color c) {
    Square ksq = pos.square<KING>(c);
    int kf = int(file_of(ksq)), kr = int(rank_of(ksq));
    int hits = 0;
    for (int df = -2; df <= 2; ++df)
    {
        for (int dr = -2; dr <= 2; ++dr)
        {
            int f = kf + df, r = kr + dr;
            if (f < 0 || f > 7 || r < 0 || r > 7)
                continue;
            Square s = Square((r << 3) | f);
            Bitboard atk = pos.attackers_to(s, ~c);
            // count piece types that matter (any non-pawn attacker counts once)
            Bitboard np = atk & ~pos.pieces(PAWN);
            if (np)
                ++hits;
            if (hits >= 2)
                return hits;
        }
    }
    return hits;
}

inline bool castled(const Position& pos, Color c) {
    Square k = pos.square<KING>(c);
    Rank r = c == WHITE ? RANK_1 : RANK_8;
    return rank_of(k) == r && (file_of(k) == FILE_G || file_of(k) == FILE_C);
}

inline int phase_moe_route(const Position& pos) {
    const int men = popcount(pos.pieces());
    if (men <= 5)
        return EX_TB;

    const auto cw = npp_side(pos, WHITE), cb = npp_side(pos, BLACK);
    // residue extras per side
    const int wn = cw.n - cb.n, wb = cw.b - cb.b, wr = cw.r - cb.r, wq = cw.q - cb.q;
    const bool asym = wn || wb || wr || wq;
    const int pawnDiff = pos.count<PAWN>(WHITE) - pos.count<PAWN>(BLACK);
    const int npp = cw.n + cw.b + cw.r + cw.q + cb.n + cb.b + cb.r + cb.q;
    const int qTot = cw.q + cb.q, rTot = cw.r + cb.r;
    const Color us = pos.side_to_move();
    const bool moverUp = us == WHITE ? pawnDiff > 0 : pawnDiff < 0;

    // residue classification on extras vector (w extras, b extras)
    auto match = [&](int a_n, int a_b, int a_r, int a_q,
                     int b_n, int b_b, int b_r, int b_q) {
        if (us == WHITE)
            return wn == a_n && wb == a_b && wr == a_r && wq == a_q
                && -wn == b_n && -wb == b_b && -wr == b_r && -wq == b_q;
        return -wn == a_n && -wb == a_b && -wr == a_r && -wq == a_q
            && wn == b_n && wb == b_b && wr == b_r && wq == b_q;
    };
    auto extras = [&](int n, int b, int r, int q, Color side) {
        int en = side == WHITE ? wn : -wn, eb = side == WHITE ? wb : -wb;
        int er = side == WHITE ? wr : -wr, eq = side == WHITE ? wq : -wq;
        return en == n && eb == b && er == r && eq == q;
    };

    if (asym)
    {
        // unordered residue pair check
        // simpler direct checks per pair (unordered)
        auto hasPair = [&](int n1, int b1, int r1, int q1,
                           int n2, int b2, int r2, int q2) {
            return (extras(n1, b1, r1, q1, WHITE) && extras(n2, b2, r2, q2, BLACK))
                || (extras(n1, b1, r1, q1, BLACK) && extras(n2, b2, r2, q2, WHITE));
        };
        if (hasPair(0, 1, 0, 0, 0, 0, 1, 0)) return EX_MVR;
        if (hasPair(2, 0, 0, 0, 0, 1, 1, 0)) return EX_RV2M;
        if (hasPair(0, 0, 1, 0, 0, 0, 0, 2)) return EX_RV2M;
        if (hasPair(1, 0, 0, 0, 0, 0, 0, 2)) return EX_RV2M;
        if (hasPair(1, 0, 0, 0, 0, 2, 0, 0)) return EX_QVMAT;
        if (hasPair(1, 0, 0, 0, 0, 1, 1, 0)) return EX_QVMAT;
        if (hasPair(1, 0, 0, 0, 0, 0, 1, 1)) return EX_QVMAT;
        if (hasPair(2, 0, 0, 0, 0, 1, 0, 0)) return EX_N2V2;  // NN v NB
        if (hasPair(2, 0, 0, 0, 0, 0, 2, 0)) return EX_N2V2;  // NN v BB
        if (!hasPair(0, 1, 0, 0, 1, 0, 0, 0))
            return moverUp ? EX_PD_UP : EX_PD_DOWN;
    }
    if (!asym && opp_bishops(pos))
        return EX_OPPB;

    if (npp <= 8 && qTot <= 2)
    {
        if (rTot && qTot) return EX_DV_QREND;
        if (rTot)         return EX_DV_REND;
        if (qTot)         return EX_DV_QEND;
        return EX_DV_CORE;
    }

    if (asym && ((extras(0, 1, 0, 0, WHITE) && extras(1, 0, 0, 0, BLACK))
              || (extras(1, 0, 0, 0, WHITE) && extras(0, 1, 0, 0, BLACK))))
        return EX_NVBLATE;

    const bool devW = minors_developed(pos, WHITE) >= 2;
    const bool devB = minors_developed(pos, BLACK) >= 2;
    if (!devW || !devB)
    {
        if (pawnDiff != 0)
            return moverUp ? EX_OP_ACCEPTOR : EX_OP_GAMBiTEER;
        const int lock = center_locked_files(pos);
        return lock >= 2 ? EX_OP_EVEN_L2 : lock == 1 ? EX_OP_EVEN_L1 : EX_OP_EVEN_L0;
    }

    const bool meC = castled(pos, us), opC = castled(pos, ~us);
    auto unsafe = [&](Color c) {
        if (king_shield(pos, c) <= 1)
            return true;
        if (king_shield(pos, c) <= 2 && king_zone_attackers(pos, c) >= 2)
            return true;
        return false;
    };
    if (unsafe(us))
        return EX_MG_UNSAFE;
    if (meC && opC)
    {
        Square kw = pos.square<KING>(WHITE), kb = pos.square<KING>(BLACK);
        if (file_of(kw) != file_of(kb))
            return EX_MG_UNSAFE;  // opposite castling = both unsafe
        return EX_MG_SAME;
    }
    if (!meC && !opC)
        return EX_MG_UNCASTLED;
    return meC ? EX_MG_MY : EX_MG_OTHER;
}

}  // namespace Stockfish

#endif  // PHASE_MOE_H_INCLUDED
'''
open(p, "w").write(new_body)
print("phase_moe23 written")
