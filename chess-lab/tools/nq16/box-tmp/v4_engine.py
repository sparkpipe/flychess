s = open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h").read()

# v4 fields
s = s.replace("""    // v3: raw board features (operator design: head sees the raw board)
    bool has_ft = false;
    u32 ft_dims = 0;
    float* ft_w = nullptr;                          // [ft_dims]""",
"""    // v3: raw board features (operator design: head sees the raw board)
    bool has_ft = false;
    u32 ft_dims = 0;
    float* ft_w = nullptr;                          // [ft_dims]

    // v4: nonlinear joint head + 14th accumulator (raw-board embedding)
    bool has_v4 = false;
    i16*  emb      = nullptr;                       // [FT_DIMS * 1024]
    float emb_scale = 1.0f;
    u32   h1 = 0, h2 = 0, in_d = 0;
    float *W1 = nullptr, *b1 = nullptr;             // [h1][in_d], [h1]
    float *W2 = nullptr, *b2 = nullptr;             // [h2][h1],  [h2]
    float *W3 = nullptr, *b3 = nullptr;             // [1][h2],   [1]""")

# v4 load: appended after v3 block
s = s.replace("""            ft_w = new float[ft_dims]();
            if (fread(ft_w, 4, ft_dims, f) != ft_dims)
                { delete[] ft_w; ft_w = nullptr; fclose(f); return false; }
            has_ft = true;
        }
        fclose(f);""",
"""            ft_w = new float[ft_dims]();
            if (fread(ft_w, 4, ft_dims, f) != ft_dims)
                { delete[] ft_w; ft_w = nullptr; fclose(f); return false; }
            has_ft = true;
        }
        if (ver == 4)
        {
            if (!has_act)
                { fprintf(stderr, "EVL fail v4 needs act\\n"); fclose(f); return false; }
            u32 eemb;
            if (fread(&eemb, 4, 1, f) != 1 || eemb != EVAL_FT_DIMS * 1024)
                { fprintf(stderr, "EVL fail emb dims\\n"); fclose(f); return false; }
            if (fread(&emb_scale, 4, 1, f) != 1)
                { fclose(f); return false; }
            emb = new i16[EVAL_FT_DIMS * 1024];
            if (fread(emb, 2, EVAL_FT_DIMS * 1024, f) != (u32)(EVAL_FT_DIMS * 1024))
                { delete[] emb; emb = nullptr; fclose(f); return false; }
            if (fread(&in_d, 4, 1, f) != 1 || in_d != EVAL_ACT_EXPERTS * 1024 + 1024 + HCE_D + EVAL_ACT_EXPERTS)
                { fprintf(stderr, "EVL fail in_d %u\\n", in_d); fclose(f); return false; }
            if (fread(&h1, 4, 1, f) != 1 || h1 > 256 || fread(&h2, 4, 1, f) != 1 || h2 > 256)
                { fclose(f); return false; }
            W1 = new float[u64(h1) * in_d](); b1 = new float[h1]();
            W2 = new float[u64(h2) * h1]();  b2 = new float[h2]();
            W3 = new float[h2]();            b3 = new float[1]();
            if (fread(W1, 4, u64(h1) * in_d, f) != u64(h1) * in_d) { fclose(f); return false; }
            if (fread(b1, 4, h1, f) != h1) { fclose(f); return false; }
            if (fread(W2, 4, u64(h2) * h1, f) != u64(h2) * h1) { fclose(f); return false; }
            if (fread(b2, 4, h2, f) != h2) { fclose(f); return false; }
            if (fread(W3, 4, h2, f) != h2) { fclose(f); return false; }
            if (fread(b3, 4, 1, f) != 1) { fclose(f); return false; }
            has_v4 = true;
        }
        fclose(f);""")

# version gate: allow 4
s = s.replace("|| ver < 1 || ver > 3)", "|| ver < 1 || ver > 4)")

# v4 apply: joint nonlinear path replaces linear paths entirely when present
s = s.replace("""    // v3: raw board features — the head's own unfiltered view""",
"""    // v4: nonlinear joint head over [13x1024 acts | 1024 raw emb | hce | onehot] + scalars
    if (head.has_v4 && expert_acts != nullptr && raw_ft != nullptr)
    {
        const u32  D  = head.act_dims;
        float      x[16 * 1024 + 64];   // 13*1024 acts + 1024 emb + hce + oh (14,398 <= 16,448)
        u32        p = 0;
        for (u32 e = 0; e < head.act_experts; e++)
        {
            const i8*    a  = expert_acts + e * D;
            const float* am = head.act_mean + e * D;
            const float* as = head.act_std + e * D;
            for (u32 d = 0; d < D; d++)
                x[p++] = (float(a[d]) - am[d]) / as[d];
        }
        // 14th accumulator: sum embedding rows of active raw features
        float raw[1024] = {0};
        for (int i = 0; i < raw_ft_count; i++)
        {
            const i16* row = head.emb + u64(raw_ft[i]) * 1024;
            for (u32 d = 0; d < 1024; d++)
                raw[d] += float(row[d]) * head.emb_scale;
        }
        for (u32 d = 0; d < 1024; d++)
            x[p++] = raw[d];
        for (int i = 0; i < EVAL_HCE_DIMS; i++)
            x[p++] = hce[i];
        for (u32 e = 0; e < head.n_experts; e++)
            x[p++] = ((int)e == routed_expert_idx ? 1.0f : 0.0f);
        const u32 IN = head.in_d;
        // layer 1
        float h1v[256];
        for (u32 j = 0; j < head.h1; j++)
        {
            float acc = head.b1[j];
            const float* w = head.W1 + u64(j) * IN;
            for (u32 i2 = 0; i2 < IN; i2++)
                acc += x[i2] * w[i2];
            h1v[j] = std::clamp(acc, 0.0f, 63.0f);
        }
        // layer 2
        float h2v[256];
        for (u32 j = 0; j < head.h2; j++)
        {
            float acc = head.b2[j];
            const float* w = head.W2 + u64(j) * head.h1;
            for (u32 i2 = 0; i2 < head.h1; i2++)
                acc += h1v[i2] * w[i2];
            h2v[j] = std::clamp(acc, 0.0f, 63.0f);
        }
        // out + scalar path (each expert's own 32->32->1 output, normalized)
        float out = head.b3[0];
        for (u32 i2 = 0; i2 < head.h2; i2++)
            out += h2v[i2] * head.W3[i2];
        float sc = 0.0f;
        const float* lw2 = head.lin_w;
        for (u32 e = 0; e < head.n_experts; e++)
            sc += norm_evals[e] * (*lw2++);
        return Value(int(out + sc + head.lin_b));
    }

    // v3: raw board features — the head's own unfiltered view""")

open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h", "w").write(s)
print("eval_head.h v4 installed")
