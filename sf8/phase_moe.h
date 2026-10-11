/*
  phase_moe.h — 14-slot MoE routing per DESIGN-MOE v1.11.

  Slots (operator-ruled taxonomy):
    0  gambit            (early asymmetric material; net trained on the
                          curated gambit pool exclusively)
    1  balanced lock_c=0     \
    2  balanced lock_c=1      \  symmetric residue, sharded by the
    3  balanced lock_c=2      /  amount of locked pawns (center files)
    4  balanced lock_c=3+    /
    5  N vs B            (residue N v B; includes 2N vs N+B — merged)
    6  N vs R            (exchange-down play included)
    7  B vs R            (exchange-down play included)
    8  R vs 2 minors     (2N / N+B / 2B vs R — merged)
    9  Q vs material     (2R / R+N / R+B vs Q — merged)
    10 opposite bishops  (symmetric bishops, opposite majority colors)
    11 dvoretsky         (6-10 men, no confrontation)
    12 exchanges         (equal-value capture available — the trade
                          decision; training is dual-membership with
                          the stable class)
    13 tablebase         (<=5 men — absolute, first check)

  2B vs 2B and 2R vs 2R are NOT experts (symmetric = balanced, ruling).
  Tactics (puzzle-trained) is OUTSIDE this cascade (dual-signal protocol).
  Matching is RESIDUE-BASED: cancel common non-pawn pieces; the imbalance
  persists in any material context (BxN mainlines). Pawns are free.
*/

#ifndef PHASE_MOE_H_INCLUDED
#define PHASE_MOE_H_INCLUDED

#include <algorithm>

#include "bitboard.h"
#include "position.h"
#include "types.h"

namespace Stockfish {

constexpr int PhaseMoESlots = 13;

enum PhaseMoESlot : int {
    // slot 0 retired: gambit (ruling 2026-09-28, augmentation data not a route)
    EX_BAL_L0 = 1,
    EX_BAL_L1 = 2,
    EX_BAL_L2 = 3,
    EX_BAL_L3 = 4,
    EX_NVSB = 5,
    EX_NVR = 6,
    EX_BVR = 7,
    EX_RV2M = 8,
    EX_QVMAT = 9,
    EX_OPPB = 10,
    EX_DVORETSKY = 11,
    EX_EXCH = 12,
    EX_TB = 13,
};

inline int bishop_majority_color(const Position& pos, Color c) {
    int sum = 0, n = 0;
    for (Bitboard b = pos.pieces(c, BISHOP); b;)
    {
        Square s = pop_lsb(b);
        sum += ((int(s) >> 3) + (int(s) & 7)) & 1;
        ++n;
    }
    if (!n)
        return -1;
    return 2 * sum > n ? 1 : 0;
}

inline int center_locked_files(const Position& pos) {
    const Bitboard wp = pos.pieces(WHITE, PAWN), bp = pos.pieces(BLACK, PAWN);
    int          lock = 0;
    for (File f = FILE_A; f <= FILE_H; ++f)
    {
        const Bitboard fw = wp & file_bb(f), fb = bp & file_bb(f);
        if (!fw || !fb)
            continue;
        Rank maxW = RANK_1, minB = RANK_8;
        for (Bitboard b = fw; b;)
        {
            Square s = pop_lsb(b);
            maxW     = std::max(maxW, rank_of(s));
        }
        for (Bitboard b = fb; b;)
        {
            Square s = pop_lsb(b);
            minB     = std::min(minB, rank_of(s));
        }
        if (minB - maxW == 1 && f >= FILE_C && f <= FILE_F)
            ++lock;
    }
    return lock;
}

// residue of the non-pawn material after canceling common pieces:
// per-side EXTRAS (non-negative counts), so matching needs no sign logic
struct Residue {
    int q[COLOR_NB], r[COLOR_NB], b[COLOR_NB], n[COLOR_NB];
};

inline Residue moe_residue(const Position& pos) {
    Residue d{};
    for (Color c : {WHITE, BLACK})
    {
        Color o   = ~c;
        d.q[c]    = std::max(0, pos.count<QUEEN>(c) - pos.count<QUEEN>(o));
        d.r[c]    = std::max(0, pos.count<ROOK>(c) - pos.count<ROOK>(o));
        d.b[c]    = std::max(0, pos.count<BISHOP>(c) - pos.count<BISHOP>(o));
        d.n[c]    = std::max(0, pos.count<KNIGHT>(c) - pos.count<KNIGHT>(o));
    }
    return d;
}

// unordered pair match: side A extras (qa,ra,ba,na) vs side B (qb,...)
inline bool moe_pair(const Residue& d, int qa, int ra, int ba, int na,
                     int qb, int rb, int bb, int nb) {
    auto eq = [&](Color A, Color B) {
        return d.q[A] == qa && d.r[A] == ra && d.b[A] == ba && d.n[A] == na
            && d.q[B] == qb && d.r[B] == rb && d.b[B] == bb && d.n[B] == nb;
    };
    return eq(WHITE, BLACK) || eq(BLACK, WHITE);
}

// equal-value capture available (same-type, or cross-minor N<->B):
// the trade-decision routing predicate for the exchanges slot
inline bool equal_trade_available(const Position& pos) {
    const Color us = pos.side_to_move(), them = ~us;
    if (pos.attacks_by<KNIGHT>(us) & pos.pieces(them, KNIGHT))
        return true;
    if (pos.attacks_by<BISHOP>(us) & pos.pieces(them, BISHOP))
        return true;
    if (pos.attacks_by<ROOK>(us) & pos.pieces(them, ROOK))
        return true;
    if (pos.attacks_by<QUEEN>(us) & pos.pieces(them, QUEEN))
        return true;
    if (pos.attacks_by<KNIGHT>(us) & pos.pieces(them, BISHOP))
        return true;  // cross-minor trades count as equal
    if (pos.attacks_by<BISHOP>(us) & pos.pieces(them, KNIGHT))
        return true;
    return false;
}

// first match wins; piece-count bounds operator-given
inline int phase_moe_route(const Position& pos) {
    const int men = popcount(pos.pieces());
    if (men <= 5)
        return EX_TB;

    const Residue d = moe_residue(pos);
    const bool symmetric =
      !(d.q[WHITE] | d.q[BLACK] | d.r[WHITE] | d.r[BLACK] | d.b[WHITE] | d.b[BLACK]
        | d.n[WHITE] | d.n[BLACK]);

    if (symmetric)
    {
        int w = bishop_majority_color(pos, WHITE), b = bishop_majority_color(pos, BLACK);
        if (w >= 0 && b >= 0 && w != b)
            return EX_OPPB;
        if (men <= 10)
            return EX_DVORETSKY;
        const int lock = center_locked_files(pos);
        return lock >= 3 ? EX_BAL_L3 : lock == 2 ? EX_BAL_L2
               : lock == 1     ? EX_BAL_L1
                              : EX_BAL_L0;
    }

    // the six residue classes (operator taxonomy, merged forms)
    if (moe_pair(d, 0, 0, 0, 1, 0, 0, 1, 0))
        return EX_NVSB;  // N v B (2N vs N+B shares the residue - merged)
    if (moe_pair(d, 0, 1, 0, 0, 0, 0, 0, 1))
        return EX_NVR;
    if (moe_pair(d, 0, 1, 0, 0, 0, 0, 1, 0))
        return EX_BVR;
    if (moe_pair(d, 0, 1, 0, 0, 0, 0, 0, 2) || moe_pair(d, 0, 1, 0, 0, 0, 0, 1, 1)
        || moe_pair(d, 0, 1, 0, 0, 0, 0, 2, 0))
        return EX_RV2M;  // R v {NN, NB, BB}
    if (moe_pair(d, 1, 0, 0, 0, 0, 2, 0, 0) || moe_pair(d, 1, 0, 0, 0, 0, 1, 0, 1)
        || moe_pair(d, 1, 0, 0, 0, 0, 1, 1, 0))
        return EX_QVMAT;  // Q v {RR, RN, RB}

    // asymmetric but unlisted residue
    if (men <= 10)
        return EX_DVORETSKY;
    if (equal_trade_available(pos))
        return EX_EXCH;  // the trade decision
    const int lock = center_locked_files(pos);
    return lock >= 3 ? EX_BAL_L3 : lock == 2 ? EX_BAL_L2
           : lock == 1     ? EX_BAL_L1
                          : EX_BAL_L0;
}

}  // namespace Stockfish

#endif  // #ifndef PHASE_MOE_H_INCLUDED
