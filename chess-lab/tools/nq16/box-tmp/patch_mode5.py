import sys
P = "/srv/workspace/flychess/src/Stockfish-act/src/search.cpp"
s = open(P).read()
a = '''    static const int dbgMode = []() {  // phase-moe debug: 0=off 1=route 2=always-slot6 3=dual
        const char* e = getenv("PHASE_MOE");
        return e ? atoi(e) : 1;
    }();
    int slot = dbgMode == 2  ? EX_DV_CORE
             : dbgMode == 0  ? 0
             : phase_moe_route(pos);'''
b = '''    static const int dbgMode = []() {  // phase-moe debug: 0=off 1=route 2=always-slot6 3=dual
        const char* e = getenv("PHASE_MOE");  // 5=route-but-use-slot0 (router-cost isolation)
        return e ? atoi(e) : 1;
    }();
    int slot = dbgMode == 2  ? EX_DV_CORE
             : dbgMode == 0  ? 0
             : phase_moe_route(pos);
    if (dbgMode == 5)
        slot = 0;'''
assert s.count(a) == 1, "anchor"
s = s.replace(a, b)
open(P, "w").write(s)
print("PATCH OK")
