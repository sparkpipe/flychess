
s = open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h").read()

s = s.replace("""constexpr int EVAL_ACT_DIMS    = 1024;  // per expert, activated transformer block
constexpr int EVAL_ACT_EXPERTS = 13;""",
"""constexpr int EVAL_ACT_DIMS    = 1024;  // per expert, activated transformer block
constexpr int EVAL_ACT_EXPERTS = 13;
constexpr int EVAL_FT_DIMS     = 86896; // raw board feature vocabulary (threats+pp)""")

s = s.replace("""    bool has_act = false;
    u32 act_experts = 0, act_dims = 0;
    float* lin_act = nullptr;                       // [act_experts * act_dims]
    alignas(64) float act_mean[EVAL_ACT_EXPERTS * EVAL_ACT_DIMS];
    alignas(64) float act_std[EVAL_ACT_EXPERTS * EVAL_ACT_DIMS];""",
"""    bool has_act = false;
    u32 act_experts = 0, act_dims = 0;
    float* lin_act = nullptr;                       // [act_experts * act_dims]
    alignas(64) float act_mean[EVAL_ACT_EXPERTS * EVAL_ACT_DIMS];
    alignas(64) float act_std[EVAL_ACT_EXPERTS * EVAL_ACT_DIMS];

    // v3: raw board features (operator design: head sees the raw board)
    bool has_ft = false;
    u32 ft_dims = 0;
    float* ft_w = nullptr;                          // [ft_dims]""")

s = s.replace("""            has_act = true;
        }
        fclose(f);""",
"""            has_act = true;
        }
        if (ver == 3)
        {
            if (!has_act)
                { fclose(f); return false; }
            if (fread(&ft_dims, 4, 1, f) != 1 || ft_dims != EVAL_FT_DIMS)
                { fclose(f); return false; }
            ft_w = new float[ft_dims]();
            if (fread(ft_w, 4, ft_dims, f) != ft_dims)
                { delete[] ft_w; ft_w = nullptr; fclose(f); return false; }
            has_ft = true;
        }
        fclose(f);""")

s = s.replace("""                             const i8* expert_acts = nullptr) {  // [13][1024] or null""",
"""                             const i8* expert_acts = nullptr,   // [13][1024] or null
                             const u16* raw_ft = nullptr,
                             int raw_ft_count = 0) {""")

s = s.replace("""    // v2: all-expert activation contributions (operator design)""",
"""    // v3: raw board features — the head's own unfiltered view
    if (head.has_ft && raw_ft != nullptr)
        for (int i = 0; i < raw_ft_count; i++)
            eval += head.ft_w[raw_ft[i]];

    // v2: all-expert activation contributions (operator design)""")

open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h", "w").write(s)
print("eval_head.h v3 patched")
