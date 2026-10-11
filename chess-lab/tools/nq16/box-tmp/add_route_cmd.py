p = "/srv/workspace/flychess/src/Stockfish-act/src/uci.cpp"
t = open(p).read()
anchor = '''        else if (token == "eval")
            engine.trace_eval();
'''
addition = '''        else if (token == "eval")
            engine.trace_eval();
        else if (token == "route")
        {  // phase-moe: print the routed expert slot for the current position
            const Position& pos = engine.get_root_position();
            const int       s   = phase_moe_route(pos);
            static const char* PhaseMoeNames[] = {
              "balanced_l0", "balanced_l1", "balanced_l2", "balanced_l3",
              "nvb", "nvr", "bvr", "rv2m", "qvmat", "oppb", "dvoretsky", "tb"};
            sync_cout << "route slot " << s << " " << PhaseMoeNames[s] << sync_endl;
        }
'''
assert anchor in t
t = t.replace(anchor, addition, 1)
open(p, "w").write(t)
print("route command added")
