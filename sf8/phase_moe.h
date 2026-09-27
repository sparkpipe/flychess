/*
  phase_moe.h — 21-specialist routing for the phase-MoE fork.

  Expert set (operator taxonomy, convo record):
    phase/structural: opening-normal, opening-gambit, mid-positional,
      mid-open, mid-queenless, endgame-dvoretsky (6-10 men),
      endgame-tablebase (<=5 men)
    material confrontations (operator list): NvB, 2NvNB, 2Bv2B, NvR, BvR,
      2NvR, NBvR, 2BvR, 2RvQ, RNvQ, RBvQ, 2Rv2R, opp-bishops,
      exchange-down

  Router = pure function of the position.  Matching reading (ruling
  pending): a confrontation class fires when the board's non-pawn
  material IS the confrontation (after canceling common pieces), so
  general middlegames with imbalances inside bigger material stay on
  the structural nets.  Pawns are free.  exchange-down = NvR/BvR core
  with the minor side holding >=1 extra pawn (the compensated,
  draw-saving side of the confrontation); NvR/BvR = uncompensated.
  Order: men<=5 TB absolute; then confrontation table; then dvoretsky
  6-10 (general practical endgames not in the table); then structural.

  Cut points PhaseMoeOpeningPly / PhaseMoeLockCut are UCI options,
  calibrated from the full-data matrix.
*/

#ifndef PHASE_MOE_H_INCLUDED
#define PHASE_MOE_H_INCLUDED

#include <algorithm>

#include "bitboard.h"
#include "position.h"
#include "types.h"

namespace Stockfish {

constexpr int PhaseMoESlots = 21;

enum PhaseMoESlot : int {
    EX_OPEN_NORMAL = 0,
    EX_OPEN_GAMBIT = 1,
    EX_MID_POSITIONAL = 2,
    EX_MID_OPEN = 3,
    EX_NVSB = 4,
    EX_2NVSNB = 5,
    EX_2BVSB = 6,
    EX_NVR = 7,
    EX_BVR = 8,
    EX_2NVR = 9,
    EX_NBVR = 10,
    EX_2BVR = 11,
    EX_2RVQ = 12,
    EX_RNVQ = 13,
    EX_RBVQ = 14,
    EX_2RV2R = 15,
    EX_OPP_BISHOPS = 16,
    EX_EXCHANGE_DOWN = 17,
    EX_MID_QUEENLESS = 18,
    EX_EG_DVORETSKY = 19,
    EX_EG_TABLEBASE = 20,
};

// Majority square color of a side's bishops, or -1 if it has none.
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

// Center files (c,d,e,f) where a white pawn stands directly below a
// black pawn — the operator's "locked" contact definition.
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

struct PhaseMoECounts {
    int q[2], r[2], b[2], n[2], p[2];  // [WHITE], [BLACK]
};

inline PhaseMoECounts moe_counts(const Position& pos) {
    PhaseMoECounts c{};
    for (Color co : {WHITE, BLACK})
    {
        c.q[co] = pos.count<QUEEN>(co);
        c.r[co] = pos.count<ROOK>(co);
        c.b[co] = pos.count<BISHOP>(co);
        c.n[co] = pos.count<KNIGHT>(co);
        c.p[co] = pos.count<PAWN>(co);
    }
    return c;
}

// opp-bishops check used inside the NvB branch (1 bishop each side)
inline bool opposite_bishops_slow(const Position& pos) {
    int w = bishop_majority_color(pos, WHITE), b = bishop_majority_color(pos, BLACK);
    return w >= 0 && b >= 0 && w != b;
}

// Confrontation table: fires only when the board's non-pawn material,
// after canceling common pieces, IS the confrontation (pawns free).
inline int moe_confrontation(const Position& pos, const PhaseMoECounts& c) {
    const int resQ[2] = {c.q[WHITE] - std::min(c.q[WHITE], c.q[BLACK]),
                         c.q[BLACK] - std::min(c.q[WHITE], c.q[BLACK])};
    const int resR[2] = {c.r[WHITE] - std::min(c.r[WHITE], c.r[BLACK]),
                         c.r[BLACK] - std::min(c.r[WHITE], c.r[BLACK])};
    const int resB[2] = {c.b[WHITE] - std::min(c.b[WHITE], c.b[BLACK]),
                         c.b[BLACK] - std::min(c.b[WHITE], c.b[BLACK])};
    const int resN[2] = {c.n[WHITE] - std::min(c.n[WHITE], c.n[BLACK]),
                         c.n[BLACK] - std::min(c.n[WHITE], c.n[BLACK])};

    // a residue matches multiset A (side w) vs multiset B (side b), either orientation
    auto match = [&](int qw, int rw, int bw, int nw, int qb, int rb, int bb, int nb) {
        return (resQ[0] == qw && resR[0] == rw && resB[0] == bw && resN[0] == nw
                && resQ[1] == qb && resR[1] == rb && resB[1] == bb && resN[1] == nb)
            || (resQ[1] == qw && resR[1] == rw && resB[1] == bw && resN[1] == nw
                && resQ[0] == qb && resR[0] == rb && resB[0] == bb && resN[0] == nb);
    };
    const int minors = c.b[0] + c.b[1] + c.n[0] + c.n[1];

    // queen confrontations: exactly one queen on the board total
    if (c.q[0] + c.q[1] == 1)
    {
        if (match(1, 0, 0, 0, 0, 2, 0, 0))
            return EX_2RVQ;  // Q vs 2R
        if (match(1, 0, 0, 0, 0, 1, 0, 1))
            return EX_RNVQ;  // Q vs R+N
        if (match(1, 0, 0, 0, 0, 1, 1, 0))
            return EX_RBVQ;  // Q vs R+B
    }

    // queens off: minor / rook confrontations
    if (c.q[0] + c.q[1] == 0)
    {
        if (match(0, 0, 0, 1, 0, 0, 1, 0))
        {
            if (minors == 2)
            {
                if (opposite_bishops_slow(pos))
                    return EX_OPP_BISHOPS;
                return EX_NVSB;  // N vs B
            }
            if (minors == 3)
                return EX_2NVSNB;  // 2N vs N+B
        }
        if (match(0, 0, 2, 0, 0, 0, 2, 0) && minors == 4)
            return EX_2BVSB;  // bishop pair vs bishop pair, no knights
        if (match(0, 1, 0, 0, 0, 0, 0, 1) || match(0, 1, 0, 0, 0, 0, 1, 0))
        {
            // R vs minor: exchange-down when the minor side holds pawn compensation
            const int pawnUpWhite = c.p[WHITE] - c.p[BLACK];
            const bool whiteIsMinorSide = resN[WHITE] + resB[WHITE] == 0;
            const int comp = whiteIsMinorSide ? -pawnUpWhite : pawnUpWhite;
            if (comp >= 1)
                return EX_EXCHANGE_DOWN;
            return resN[0] == 1 || resN[1] == 1 ? EX_NVR : EX_BVR;
        }
        if (match(0, 1, 0, 0, 0, 0, 0, 2))
            return EX_2NVR;  // R vs 2N
        if (match(0, 1, 0, 0, 0, 0, 1, 1))
            return EX_NBVR;  // R vs N+B
        if (match(0, 1, 0, 0, 0, 0, 2, 0))
            return EX_2BVR;  // R vs 2B
        if (resQ[0] == 0 && resQ[1] == 0 && resR[0] == 0 && resR[1] == 0
            && resB[0] == 0 && resB[1] == 0 && resN[0] == 0 && resN[1] == 0
            && c.r[0] == 2 && c.r[1] == 2)
            return EX_2RV2R;  // two rooks each, nothing else non-pawn
    }

    return -1;  // no confrontation class
}

// First match wins.  men<=5 absolute (exact knowledge); then the
// confrontation table; then dvoretsky 6-10; then structural splits.
inline int phase_moe_route(const Position& pos, int openingPly, int lockCut, bool) {
    const int men = popcount(pos.pieces());
    if (men <= 5)
        return EX_EG_TABLEBASE;

    const PhaseMoECounts c     = moe_counts(pos);
    const int            confl = moe_confrontation(pos, c);
    if (confl >= 0)
        return confl;

    if (men <= 10)
        return EX_EG_DVORETSKY;

    if (c.q[0] + c.q[1] == 0)
        return EX_MID_QUEENLESS;

    const bool asym = c.n[WHITE] != c.n[BLACK] || c.b[WHITE] != c.b[BLACK]
                   || c.r[WHITE] != c.r[BLACK] || c.q[WHITE] != c.q[BLACK];
    if (int(pos.game_ply()) < openingPly)
        return asym ? EX_OPEN_GAMBIT : EX_OPEN_NORMAL;

    // general middlegames (including unlisted imbalances inside bigger
    // material — the nets see full position) route by pawn contact
    return center_locked_files(pos) >= lockCut ? EX_MID_POSITIONAL : EX_MID_OPEN;
}

}  // namespace Stockfish

#endif  // #ifndef PHASE_MOE_H_INCLUDED
