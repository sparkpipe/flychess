/*
  eval_head.h — stacked MoE using expert SCALAR EVALS as features.
  Each expert runs its FULL evaluation (not just the transformer),
  producing 13 scalar opinions. The head combines those + HCE features.
*/
#ifndef EVAL_HEAD_H_INCLUDED
#define EVAL_HEAD_H_INCLUDED

#include <cstring>
#include <algorithm>
#include "phase_moe.h"

namespace Stockfish {

constexpr u32 EVAL_HEAD_MAGIC = 0x45564C48; // "EVLH"
constexpr int EVAL_HCE_DIMS = 36;
constexpr int EVAL_MAX_EXPERTS = 14;
constexpr int EVAL_MAX_DOMAINS = 13;

struct EvalHead {
    bool loaded = false;
    u32 n_experts, hce_dims, n_domains;

    // gates[domain][expert] — learned per-domain expert trust
    alignas(64) float gates[EVAL_MAX_DOMAINS][EVAL_MAX_EXPERTS];
    // linear weights for [evals || hce || onehot]
    alignas(64) float lin_w[EVAL_MAX_EXPERTS + EVAL_HCE_DIMS + EVAL_MAX_EXPERTS];
    float lin_b;
    // per-expert normalization (from training)
    alignas(64) float eval_mean[EVAL_MAX_EXPERTS];
    alignas(64) float eval_std[EVAL_MAX_EXPERTS];

    bool load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) return false;
        u32 magic;
        if (fread(&magic, 4, 1, f) != 1 || magic != EVAL_HEAD_MAGIC)
            { fclose(f); return false; }
        u32 ver;
        if (fread(&ver, 4, 1, f) != 1 || ver != 1)
            { fclose(f); return false; }
        if (fread(&n_experts, 4, 1, f) != 1 || n_experts > EVAL_MAX_EXPERTS)
            { fclose(f); return false; }
        if (fread(&hce_dims, 4, 1, f) != 1 || hce_dims > EVAL_HCE_DIMS)
            { fclose(f); return false; }
        if (fread(&n_domains, 4, 1, f) != 1 || n_domains > EVAL_MAX_DOMAINS)
            { fclose(f); return false; }
        if (fread(gates, 4, n_domains * n_experts, f) != n_domains * n_experts)
            { fclose(f); return false; }
        u32 lw_size = n_experts + hce_dims + n_experts;
        if (fread(lin_w, 4, lw_size, f) != lw_size)
            { fclose(f); return false; }
        if (fread(&lin_b, 4, 1, f) != 1)
            { fclose(f); return false; }
        if (fread(eval_mean, 4, n_experts, f) != n_experts)
            { fclose(f); return false; }
        if (fread(eval_std, 4, n_experts, f) != n_experts)
            { fclose(f); return false; }
        fclose(f);
        loaded = true;
        return true;
    }
};

// Compute HCE features (must match train_head_evals.py exactly)
inline void compute_hce(const Position& pos, float* out) {
    int idx = 0;
    // piece counts (10)
    constexpr int pv[] = {1, 3, 3, 5, 9};
    for (int pi = 0; pi < 5; pi++) {
        int wc = 0, bc = 0;
        switch (pi) {
        case 0: wc = pos.count<PAWN>(WHITE); bc = pos.count<PAWN>(BLACK); break;
        case 1: wc = pos.count<KNIGHT>(WHITE); bc = pos.count<KNIGHT>(BLACK); break;
        case 2: wc = pos.count<BISHOP>(WHITE); bc = pos.count<BISHOP>(BLACK); break;
        case 3: wc = pos.count<ROOK>(WHITE); bc = pos.count<ROOK>(BLACK); break;
        case 4: wc = pos.count<QUEEN>(WHITE); bc = pos.count<QUEEN>(BLACK); break;
        }
        out[idx++] = float(wc * pv[pi]) / 9.0f;
        out[idx++] = float(bc * pv[pi]) / 9.0f;
    }
    // stm (1)
    out[idx++] = (pos.side_to_move() == WHITE) ? 1.0f : -1.0f;
    // men (1)
    out[idx++] = float(popcount(pos.pieces())) / 32.0f;
    // move number (1)
    out[idx++] = float(pos.game_ply() / 2 + 1) / 100.0f;
    // castling (4)
    out[idx++] = pos.can_castle(WHITE_OO) ? 1.0f : 0.0f;
    out[idx++] = pos.can_castle(WHITE_OOO) ? 1.0f : 0.0f;
    out[idx++] = pos.can_castle(BLACK_OO) ? 1.0f : 0.0f;
    out[idx++] = pos.can_castle(BLACK_OOO) ? 1.0f : 0.0f;
    // en passant (1)
    out[idx++] = (pos.ep_square() != SQ_NONE) ? 1.0f : 0.0f;
    // pawn files per side (8+1 each = 18)
    for (Color c : {WHITE, BLACK}) {
        int fc[8] = {0};
        Bitboard pawns = pos.pieces(c, PAWN);
        while (pawns) {
            Square s = pop_lsb(pawns);
            fc[int(file_of(s))]++;
        }
        for (int fi = 0; fi < 8; fi++)
            out[idx++] = float(fc[fi]) / 2.0f;
        int doubled = 0;
        for (int fi = 0; fi < 8; fi++)
            if (fc[fi] >= 2) doubled++;
        out[idx++] = float(doubled) / 4.0f;
    }
}

// Apply the eval-based head: 13 scalar evals + HCE → final eval
inline Value apply_eval_head(const EvalHead& head,
                             const Value* expert_evals,
                             const Position& pos,
                             int domain_idx,
                             int routed_expert_idx) {
    if (!head.loaded) return VALUE_ZERO;

    alignas(64) float hce[EVAL_HCE_DIMS];
    compute_hce(pos, hce);

    // normalize evals
    alignas(64) float norm_evals[EVAL_MAX_EXPERTS];
    for (u32 e = 0; e < head.n_experts; e++)
        norm_evals[e] = (float(expert_evals[e]) - head.eval_mean[e])
                      / head.eval_std[e];

    // gate by domain
    const float* gates_row = head.gates[domain_idx];
    float eval = head.lin_b;
    const float* lw = head.lin_w;
    for (u32 e = 0; e < head.n_experts; e++)
        eval += norm_evals[e] * gates_row[e] * (*lw++);
    for (int i = 0; i < EVAL_HCE_DIMS; i++)
        eval += hce[i] * (*lw++);
    for (u32 e = 0; e < head.n_experts; e++)
        eval += ((int)e == routed_expert_idx ? 1.0f : 0.0f) * (*lw++);

    return Value(int(eval));
}

} // namespace Stockfish

#endif
