
s = open("/srv/workspace/flychess/src/Stockfish-act/src/search.cpp").read()

s = s.replace("""        // phase-moe v2: capture ALL experts stm activated blocks (13x1024)
        alignas(64) i8 expertActs[PhaseMoESlots][1024];
        const bool wantActs = evalHead->has_act;""",
"""        // phase-moe v2: capture ALL experts stm activated blocks (13x1024)
        alignas(64) i8 expertActs[PhaseMoESlots][1024];
        const bool wantActs = evalHead->has_act;

        // phase-moe v3: raw board active feature indices (stm perspective)
        u16 rawFt[512];
        int rawFtCount = 0;
        if (evalHead->has_ft)
        {
            const Color us = pos.side_to_move();
            NNUE::ThreatFeatureSet::IndexList tlist;
            NNUE::ThreatFeatureSet::append_active_indices(us, pos, tlist);
            for (auto idx : tlist)
                rawFt[rawFtCount++] = u16(idx);
            NNUE::PairFeatureSet::IndexList plist;
            NNUE::PairFeatureSet::append_active_indices(us, pos, plist);
            for (auto idx : plist)
                rawFt[rawFtCount++] = u16(idx);
        }""")

s = s.replace("""        return apply_eval_head(*evalHead, expertVals, pos,
                              slot, slot,
                              wantActs ? &expertActs[0][0] : nullptr);""",
"""        return apply_eval_head(*evalHead, expertVals, pos,
                              slot, slot,
                              wantActs ? &expertActs[0][0] : nullptr,
                              evalHead->has_ft ? rawFt : nullptr,
                              rawFtCount);""")

open("/srv/workspace/flychess/src/Stockfish-act/src/search.cpp", "w").write(s)
print("search.cpp v3 wired")
