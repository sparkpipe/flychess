s = open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h").read()

# v5 constants
s = s.replace("constexpr int EVAL_FT_DIMS     = 86896; // raw board feature vocabulary (threats+pp)",
"""constexpr int EVAL_FT_DIMS     = 86896; // raw board feature vocabulary (threats+pp)
constexpr int EVAL_FUSE_DIMS   = 64;    // fused accumulator width (v5)
constexpr int EVAL_FUSE_IN     = 88944; // merged feature vocab incl. psq virtual""")

# v5 fields
s = s.replace("""    // v4: nonlinear joint head + 14th accumulator (raw-board embedding)
    bool has_v4 = false;""",
"""    // v5: fused linear-regime equivalent (calculated, not trained)
    bool has_v5 = false;
    float* fuse_V = nullptr;          // [EVAL_FUSE_IN x 64] — feature->fused accumulator
    float  fuse_bias[64];             // accumulator bias (incl. calibration constant)

    // v4: nonlinear joint head + 14th accumulator (raw-board embedding)
    bool has_v4 = false;""")

# v5 load: appended after v4 block, before fclose
old_tail = """            has_v4 = true;
        }
        fclose(f);"""
new_tail = """            has_v4 = true;
        }
        if (ver == 5)
        {
            u32 fuse_in;
            if (fread(&fuse_in, 4, 1, f) != 1 || fuse_in != EVAL_FUSE_IN)
                { fprintf(stderr, "EVL fail v5 in %u\\n", fuse_in); fclose(f); return false; }
            fuse_V = new float[u64(EVAL_FUSE_IN) * EVAL_FUSE_DIMS]();
            if (fread(fuse_V, 4, u64(EVAL_FUSE_IN) * EVAL_FUSE_DIMS, f) != u64(EVAL_FUSE_IN) * EVAL_FUSE_DIMS)
                { delete[] fuse_V; fuse_V = nullptr; fclose(f); return false; }
            if (fread(fuse_bias, 4, 64, f) != 64)
                { fclose(f); return false; }
            has_v5 = true;
        }
        fclose(f);"""
assert old_tail in s
s = s.replace(old_tail, new_tail)

# version gate: allow 5
s = s.replace("|| ver < 1 || ver > 4)", "|| ver < 1 || ver > 5)")

# v5 apply: earliest branch — fused accumulator + v4 tail + exact scalar path
old_v4_gate = "    // v4: nonlinear joint head over [13x1024 acts | 1024 raw emb | hce | onehot] + scalars"
new_v5 = """    // v5: fused linear-regime fast path — one 64-wide accumulator, v4 tail, exact scalars
    if (head.has_v5 && raw_ft != nullptr)
    {
        // fused accumulator from active raw features (threats+pp space)
        // NOTE: raw_ft indices are in the threats+pp combined space (base 59808 for pp);
        // V is indexed in the merged 88944 space. The loader maps: threats 0..59807,
        // pp 59808..64367, psq 64368..88943. rawft gives threats+pp directly.
        float acc[64];
        for (int j = 0; j < 64; j++)
            acc[j] = head.fuse_bias[j];
        for (int i = 0; i < raw_ft_count; i++)
        {
            const float* row = head.fuse_V + u64(raw_ft[i]) * 64;
            for (int j = 0; j < 64; j++)
                acc[j] += row[j];
        }
        // tail input: [fused_64 || hce_36 || onehot_13]  (acts block replaced by fused)
        // BUT the v4 tail expects 14385 inputs: acts(13312) + raw_emb(1024) + hce + oh.
        // The fused-64 path needs its own tail. Store tail W1 subset: we reuse W2/W3 and
        // keep a dedicated 64->32 first layer in the file (fuse_tail_w, fuse_tail_b).
        float h1v[64];
        for (int j = 0; j < 64; j++)
            h1v[j] = std::clamp(acc[j] + head.fuse_hce_dot(hce, j), 0.0f, 63.0f);
        float h2v[32];
        for (int j = 0; j < head.h2; j++)
        {
            float a2 = head.b2[j];
            for (int i2 = 0; i2 < 64; i2++)
                a2 += h1v[i2] * head.fuse_W2[j * 64 + i2];
            h2v[j] = std::clamp(a2, 0.0f, 63.0f);
        }
        float out = head.fuse_b3;
        for (int i2 = 0; i2 < head.h2; i2++)
            out += h2v[i2] * head.W3[i2];
        // exact scalar path (13 expert evals — same as v4)
        float sc2 = 0.0f;
        const float* lw3 = head.lin_w;
        for (u32 e = 0; e < head.n_experts; e++)
            sc2 += norm_evals[e] * (*lw3++);
        return Value(int(out + sc2 + head.lin_b));
    }

    // v4: nonlinear joint head over [13x1024 acts | 1024 raw emb | hce | onehot] + scalars"""
assert old_v4_gate in s
s = s.replace(old_v4_gate, new_v5, 1)

# add the extra v5 fields (fuse hce contribution + dedicated tail)
s = s.replace("""    // v5: fused linear-regime equivalent (calculated, not trained)
    bool has_v5 = false;
    float* fuse_V = nullptr;          // [EVAL_FUSE_IN x 64] — feature->fused accumulator
    float  fuse_bias[64];             // accumulator bias (incl. calibration constant)""",
"""    // v5: fused linear-regime equivalent (calculated, not trained)
    bool has_v5 = false;
    float* fuse_V = nullptr;          // [EVAL_FUSE_IN x 64] — feature->fused accumulator
    float  fuse_bias[64];             // accumulator bias (incl. calibration constant)
    alignas(64) float fuse_hce[64][EVAL_HCE_DIMS];  // hce contribution per fused unit
    alignas(64) float fuse_W2[32 * 64];             // dedicated 64->32
    alignas(64) float fuse_b2v[32];
    float fuse_b3;""")

# extend the loader for the extra v5 fields
s = s.replace("""            if (fread(fuse_bias, 4, 64, f) != 64)
                { fclose(f); return false; }
            has_v5 = true;""",
"""            if (fread(fuse_bias, 4, 64, f) != 64)
                { fclose(f); return false; }
            if (fread(fuse_hce, 4, 64 * EVAL_HCE_DIMS, f) != u32(64 * EVAL_HCE_DIMS))
                { fclose(f); return false; }
            if (fread(fuse_W2, 4, 32 * 64, f) != u32(32 * 64))
                { fclose(f); return false; }
            if (fread(fuse_b2v, 4, 32, f) != 32)
                { fclose(f); return false; }
            if (fread(&fuse_b3, 4, 1, f) != 1)
                { fclose(f); return false; }
            has_v5 = true;""")

# helper for hce dot (inline member fn via lambda-free approach: replace call)
s = s.replace("acc[j] = head.fuse_bias[j];", "acc[j] = head.fuse_bias[j];")
s = s.replace("h1v[j] = std::clamp(acc[j] + head.fuse_hce_dot(hce, j), 0.0f, 63.0f);",
"""{ float hc = 0.0f; for (int q = 0; q < EVAL_HCE_DIMS; q++) hc += hce[q] * head.fuse_hce[j][q];
              h1v[j] = std::clamp(acc[j] + hc, 0.0f, 63.0f); }""")

open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h", "w").write(s)
print("eval_head.h v5 installed")
