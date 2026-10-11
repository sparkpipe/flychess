s = open("/srv/workspace/flychess/src/Stockfish-act/src/search.cpp").read()
r1o = "        Value expertVals[PhaseMoESlots] = { VALUE_ZERO };"
r1n = """        Value expertVals[PhaseMoESlots] = { VALUE_ZERO };
        // phase-moe v2: capture ALL experts stm activated blocks (13x1024)
        alignas(64) i8 expertActs[PhaseMoESlots][1024];
        const bool wantActs = evalHead->has_act;"""
assert r1o in s, "r1"
s = s.replace(r1o, r1n, 1)

r2o = "            expertVals[s] = (*networks[s])[numaAccessToken].evaluate(\n                pos, accumulatorStacks[s], refreshTables[s]);"
r2n = """            expertVals[s] = (*networks[s])[numaAccessToken].evaluate_capture(
                pos, accumulatorStacks[s], refreshTables[s],
                wantActs ? expertActs[s].data() : nullptr);"""
assert r2o in s, "r2"
s = s.replace(r2o, r2n, 1)

r3o = "        return apply_eval_head(*evalHead, expertVals, pos,"
assert r3o in s, "r3"
import re
m = re.search(r"(        return apply_eval_head\(\*evalHead, expertVals, pos,\n\s*slot, slot\);)", s)
assert m, "r3 full"
s = s.replace(m.group(1),
"""        return apply_eval_head(*evalHead, expertVals, pos,
                              slot, slot,
                              wantActs ? &expertActs[0][0] : nullptr);""", 1)
open("/srv/workspace/flychess/src/Stockfish-act/src/search.cpp", "w").write(s)
print("search.cpp wired")
