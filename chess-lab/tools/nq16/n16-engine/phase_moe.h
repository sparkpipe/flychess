/*
  phase_moe.h — n16 taxonomy (operator rulings 2026-10-05/06, EXPERTS16.md).

  Slots (dense 0..15), check order = priority:
    0  tb            total pieces <= 5
    1  mvr           R v one minor (the exchange)
    2  rv2m          R v exactly two minors
    3  qvmat         Q v two or three pieces (R/N/B)
    4  nvb           N v B family (incl. 2N v 2B)
    5  piece_down    any other material imbalance
    6  oppb          equal material, opposite-color bishops (1B v 1B)
    7  dv_Q          sparse endgames (<=8 npp, <=2Q) with queens
    8  dv_R          ... rooks, no queens
    9  dv_rest       ... minors/pawns only
   10  op_pawnimb    fullmove < 15, unequal pawns (equal pieces)
   11  op_even_l0    fm<15, even, 0 locked center files
   12  op_even_l1    ... 1 lock
   13  op_even_l2p   ... 2+ locks
   14  mg_unsafe     opposite castling or own king shield <= 1
   15  mg_safe       all other middlegames

  No side-to-move keys. No catch-all domain: every asymmetric material
  vector resolves to a named pattern or piece_down.
*/

#ifndef PHASE_MOE_H_INCLUDED
#define PHASE_MOE_H_INCLUDED

#include <algorithm>

#include "bitboard.h"
#include "position.h"
#include "types.h"

namespace Stockfish {

constexpr int PhaseMoESlots = 16;

enum PhaseMoESlot : int {
    EX_TB = 0, EX_MVR = 1, EX_RV2M = 2, EX_QVMAT = 3, EX_NVB = 4,
    EX_PDOWN = 5, EX_OPPB = 6, EX_DV_Q = 7, EX_DV_R = 8, EX_DV_REST = 9,
    EX_OP_PAWNIMB = 10, EX_OP_EVEN_L0 = 11, EX_OP_EVEN_L1 = 12, EX_OP_EVEN_L2P = 13,
    EX_MG_UNSAFE = 14, EX_MG_SAFE = 15,
};

static_assert(EX_TB == 0 && EX_MG_SAFE == PhaseMoESlots - 1,
              "phase-moe slot ids must be dense 0..PhaseMoESlots-1");

inline int center_locked_files(const Position& pos) {
    const Bitboard wp = pos.pieces(WHITE, PAWN), bp = pos.pieces(BLACK, PAWN);
    int lock = 0;
    for (File f = FILE_C; f <= FILE_F; ++f)
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
        if (minB - maxW == 1)
            ++lock;
    }
    return lock;
}

struct NppSide { int n, b, r, q; };

inline NppSide npp_side(const Position& pos, Color c) {
    return {pos.count<KNIGHT>(c), pos.count<BISHOP>(c), pos.count<ROOK>(c), pos.count<QUEEN>(c)};
}

inline bool opp_bishops(const Position& pos) {
    if (pos.count<BISHOP>(WHITE) != 1 || pos.count<BISHOP>(BLACK) != 1)
        return false;   // exactly 1B v 1B (measured: no multi-bishop leakage)
    int wOdd = 0, bOdd = 0;
    for (Bitboard b = pos.pieces(WHITE, BISHOP); b;)
    {
        Square s = pop_lsb(b);
        wOdd += ((int(s) >> 3) + (int(s) & 7)) & 1;
    }
    for (Bitboard b = pos.pieces(BLACK, BISHOP); b;)
    {
        Square s = pop_lsb(b);
        bOdd += ((int(s) >> 3) + (int(s) & 7)) & 1;
    }
    return wOdd != bOdd;
}

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
    const int wn = cw.n - cb.n, wb = cw.b - cb.b, wr = cw.r - cb.r, wq = cw.q - cb.q;
    const bool asym = wn || wb || wr || wq;

    if (asym)
    {
        auto pat = [&](int n, int b, int r, int q) {
            return (wn == n && wb == b && wr == r && wq == q)
                || (wn == -n && wb == -b && wr == -r && wq == -q);
        };
        if (pat(1, 0, -1, 0) || pat(0, 1, -1, 0))
            return EX_MVR;
        if (pat(2, 0, -1, 0) || pat(0, 2, -1, 0) || pat(1, 1, -1, 0))
            return EX_RV2M;
        static const int qv[16][3] = {
            {2,0,0},{0,2,0},{0,0,2},{1,1,0},{1,0,1},{0,1,1},
            {3,0,0},{0,3,0},{0,0,3},{2,1,0},{2,0,1},{1,2,0},
            {0,2,1},{1,0,2},{0,1,2},{1,1,1}};
        for (auto& t : qv)
            if (pat(t[0], t[1], t[2], -1))
                return EX_QVMAT;
        if (pat(1, -1, 0, 0) || pat(2, -2, 0, 0))
            return EX_NVB;
        return EX_PDOWN;
    }

    if (opp_bishops(pos))
        return EX_OPPB;

    const int npp = cw.n + cw.b + cw.r + cw.q + cb.n + cb.b + cb.r + cb.q;
    const int qTot = cw.q + cb.q, rTot = cw.r + cb.r;
    if (npp <= 8 && qTot <= 2)
    {
        if (qTot)
            return EX_DV_Q;
        if (rTot)
            return EX_DV_R;
        return EX_DV_REST;
    }

    const int fm = pos.game_ply() / 2 + 1;
    if (fm < 15)
    {
        if (pos.count<PAWN>(WHITE) != pos.count<PAWN>(BLACK))
            return EX_OP_PAWNIMB;
        const int lock = center_locked_files(pos);
        return lock >= 2 ? EX_OP_EVEN_L2P : lock == 1 ? EX_OP_EVEN_L1 : EX_OP_EVEN_L0;
    }

    const bool meC = castled(pos, WHITE), opC = castled(pos, BLACK);
    if (meC && opC)
    {
        Square kw = pos.square<KING>(WHITE), kb = pos.square<KING>(BLACK);
        if (std::abs(int(file_of(kw)) - int(file_of(kb))) >= 3)
            return EX_MG_UNSAFE;   // opposite castling
        if (king_shield(pos, WHITE) <= 1 || king_shield(pos, BLACK) <= 1)
            return EX_MG_UNSAFE;
        return EX_MG_SAFE;
    }
    if (king_shield(pos, WHITE) <= 1 || king_shield(pos, BLACK) <= 1)
        return EX_MG_UNSAFE;
    return EX_MG_SAFE;
}

}  // namespace Stockfish

#endif  // PHASE_MOE_H_INCLUDED
