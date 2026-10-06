/*
  stack_head.h — stacked MoE evaluation: all experts' transformer outputs
  fed as features into a learned linear head with per-domain gates.

  The two-step process:
    Step 1: run every loaded expert's feature transformer on the position
            → N_EXP × 1024 activation values
    Step 2: gate by domain, concatenate with HCE features, dot with
            head weights → eval (replaces single-expert routing)

  Head weights loaded from a binary file (StackHead UCI option).
  File format (little-endian):
    [u32 magic=0x53544B48] [u32 version=1] [u32 n_experts] [u32 act_dims]
    [u32 hce_dims] [u32 n_domains]
    [f32 gates[n_domains][n_experts]]
    [f32 linear_w[act_dims*n_experts + hce_dims + n_experts]]
    [f32 linear_b]
*/
#ifndef STACK_HEAD_H_INCLUDED
#define STACK_HEAD_H_INCLUDED

#include <cstring>
#include <algorithm>
#include "phase_moe.h"

namespace Stockfish {

constexpr u32 STACK_HEAD_MAGIC = 0x53544B48; // "STKH"
constexpr int STACK_HCE_DIMS = 36;
constexpr int STACK_MAX_ACT_DIMS = 1024;
constexpr int STACK_MAX_EXPERTS = 14;
constexpr int STACK_MAX_DOMAINS = 13;

struct StackHead {
    bool loaded = false;
    u32 n_experts = 0;
    u32 act_dims = 0;
    u32 n_domains = 0;

    alignas(64) float gates[STACK_MAX_DOMAINS][STACK_MAX_EXPERTS];
    alignas(64) float linear_w[STACK_MAX_ACT_DIMS * STACK_MAX_EXPERTS
                              + STACK_HCE_DIMS + STACK_MAX_EXPERTS];
    float linear_b = 0.0f;
    alignas(64) float act_scale[STACK_MAX_EXPERTS];
    alignas(64) float act_offset[STACK_MAX_EXPERTS];

    bool load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f)
            return false;

        u32 magic, version;
        if (fread(&magic, 4, 1, f) != 1 || magic != STACK_HEAD_MAGIC)
            { fclose(f); return false; }
        if (fread(&version, 4, 1, f) != 1 || version != 1)
            { fclose(f); return false; }
        if (fread(&n_experts, 4, 1, f) != 1 || n_experts > STACK_MAX_EXPERTS)
            { fclose(f); return false; }
        if (fread(&act_dims, 4, 1, f) != 1 || act_dims > STACK_MAX_ACT_DIMS)
            { fclose(f); return false; }
        fread(&hce_dims_file, 4, 1, f);
        fread(&n_domains, 4, 1, f);
        if (n_domains > STACK_MAX_DOMAINS)
            { fclose(f); return false; }

        if (fread(gates, 4, n_domains * n_experts, f) != n_domains * n_experts)
            { fclose(f); return false; }
        u32 lw_size = act_dims * n_experts + STACK_HCE_DIMS + n_experts;
        if (fread(linear_w, 4, lw_size, f) != lw_size)
            { fclose(f); return false; }
        if (fread(&linear_b, 4, 1, f) != 1)
            { fclose(f); return false; }

        // normalization constants
        if (fread(act_scale, 4, n_experts, f) != n_experts)
            { fclose(f); return false; }
        if (fread(act_offset, 4, n_experts, f) != n_experts)
            { fclose(f); return false; }

        fclose(f);
        loaded = true;
        return true;
    }

    u32 hce_dims_file = STACK_HCE_DIMS;
};

// HCE features computed from position (must match train_head.py exactly)
// Returns STACK_HCE_DIMS floats
inline void compute_hce(const Position& pos, float* out) {
    int idx = 0;
    // piece counts per type per side (10 values)
    for (PieceType pt : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}) {
        int v = (pt == PAWN) ? 1 : (pt == KNIGHT || pt == BISHOP) ? 3
              : (pt == ROOK) ? 5 : 9;
        out[idx++] = float(pos.count<PAWN>(WHITE) * v)
                   / 9.0f; // placeholder — will fix below
    }
    // Actually let me redo this properly:
    idx = 0;
    static constexpr int piece_vals[] = {1, 3, 3, 5, 9}; // P N B R Q
    for (int pi = 0; pi < 5; pi++) {
        switch (pi) {
        case 0:
            out[idx++] = float(pos.count<PAWN>(WHITE)) / 8.0f;
            out[idx++] = float(pos.count<PAWN>(BLACK)) / 8.0f;
            break;
        case 1:
            out[idx++] = float(pos.count<KNIGHT>(WHITE) * 3) / 9.0f;
            out[idx++] = float(pos.count<KNIGHT>(BLACK) * 3) / 9.0f;
            break;
        case 2:
            out[idx++] = float(pos.count<BISHOP>(WHITE) * 3) / 9.0f;
            out[idx++] = float(pos.count<BISHOP>(BLACK) * 3) / 9.0f;
            break;
        case 3:
            out[idx++] = float(pos.count<ROOK>(WHITE) * 5) / 9.0f;
            out[idx++] = float(pos.count<ROOK>(BLACK) * 5) / 9.0f;
            break;
        case 4:
            out[idx++] = float(pos.count<QUEEN>(WHITE) * 9) / 9.0f;
            out[idx++] = float(pos.count<QUEEN>(BLACK) * 9) / 9.0f;
            break;
        }
    }
    // stm (1)
    out[idx++] = (pos.side_to_move() == WHITE) ? 1.0f : -1.0f;
    // men count (1)
    out[idx++] = float(popcount(pos.pieces())) / 32.0f;
    // move number (1)
    out[idx++] = float(pos.game_ply() / 2 + 1) / 100.0f;
    // castling rights (4)
    out[idx++] = pos.can_castle(WHITE_OO) ? 1.0f : 0.0f;
    out[idx++] = pos.can_castle(WHITE_OOO) ? 1.0f : 0.0f;
    out[idx++] = pos.can_castle(BLACK_OO) ? 1.0f : 0.0f;
    out[idx++] = pos.can_castle(BLACK_OOO) ? 1.0f : 0.0f;
    // en passant (1)
    out[idx++] = (pos.ep_square() != SQ_NONE) ? 1.0f : 0.0f;
    // pawn files per side (8+1 each = 18)
    for (Color c : {WHITE, BLACK}) {
        int file_counts[8] = {0};
        Bitboard pawns = pos.pieces(c, PAWN);
        while (pawns) {
            Square s = pop_lsb(pawns);
            file_counts[int(file_of(s))]++;
        }
        for (int f = 0; f < 8; f++)
            out[idx++] = float(file_counts[f]) / 2.0f;
        int doubled = 0;
        for (int f = 0; f < 8; f++)
            if (file_counts[f] >= 2) doubled++;
        out[idx++] = float(doubled) / 4.0f;
    }
    // total: 10+1+1+1+4+1+9+9 = 36
}

// Apply the stacked head: activations from all experts + HCE → eval
// activations: [n_experts][act_dims] raw transformer outputs
// domain_idx: which routing domain this position belongs to
// routed_expert_idx: which expert the router would pick (for one-hot)
inline Value apply_stack_head(const StackHead& head,
                              const Eval::NNUE::TransformedFeatureType* activations,
                              const Position& pos,
                              int domain_idx,
                              int routed_expert_idx) {
    if (!head.loaded)
        return VALUE_ZERO;

    alignas(64) float hce[STACK_HCE_DIMS];
    compute_hce(pos, hce);

    // dot product accumulator
    float eval = head.linear_b;
    const float* lw = head.linear_w;
    const float* gates_row = head.gates[domain_idx];

    // gated activations
    for (u32 e = 0; e < head.n_experts; e++) {
        float gate = gates_row[e];
        const auto* act = activations + e * head.act_dims;
        // normalize + calibrate
        float scale = head.act_scale[e];
        float offset = head.act_offset[e];
        for (u32 d = 0; d < head.act_dims; d++) {
            float a = (float(act[d]) - offset) * scale;
            eval += a * gate * (*lw++);
        }
    }
    // HCE features
    for (int i = 0; i < STACK_HCE_DIMS; i++)
        eval += hce[i] * (*lw++);
    // routed one-hot
    for (u32 e = 0; e < head.n_experts; e++)
        eval += ((int)e == routed_expert_idx ? 1.0f : 0.0f) * (*lw++);

    return Value(int(eval));
}

} // namespace Stockfish

#endif // STACK_HEAD_H_INCLUDED
