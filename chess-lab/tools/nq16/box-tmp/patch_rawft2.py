# engine.h: declare rawft
s = open("/srv/workspace/flychess/src/Stockfish-act/src/engine.h").read()
if "rawft" not in s:
    s = s.replace("    void trace_eval() const;",
                  "    void trace_eval() const;\n    void rawft(const std::string& fen) const;  // phase-moe: dump raw active feature indices")
    open("/srv/workspace/flychess/src/Stockfish-act/src/engine.h", "w").write(s)
    print("engine.h ok")
else:
    print("engine.h already patched")

# engine.cpp: implement rawft
s = open("/srv/workspace/flychess/src/Stockfish-act/src/engine.cpp").read()
if "Engine::rawft" not in s:
    anchor = "const OptionsMap& Engine::get_options() const { return options; }"
    impl = (
        'void Engine::rawft(const std::string& fen) const {\n'
        '    StateListPtr states(new std::deque<StateInfo>(1));\n'
        '    Position     p;\n'
        '    std::string  f = fen;\n'
        '    while (!f.empty() && f.front() == \' \')\n'
        '        f.erase(f.begin());\n'
        '    p.set(f.empty() ? pos.fen() : f, options["UCI_Chess960"], &states->back());\n'
        '    const Color  us = p.side_to_move();\n'
        '    Eval::NNUE::ThreatFeatureSet::IndexList tlist;\n'
        '    Eval::NNUE::ThreatFeatureSet::append_active_indices(us, p, tlist);\n'
        '    Eval::NNUE::PairFeatureSet::IndexList plist;\n'
        '    Eval::NNUE::PairFeatureSet::append_active_indices(us, p, plist);\n'
        '    std::string out;\n'
        '    for (auto i : tlist)\n'
        '        out += std::to_string(u32(i)) + " ";\n'
        '    for (auto i : plist)\n'
        '        out += std::to_string(u32(i)) + " ";\n'
        '    sync_cout << "rawft " << out << sync_endl;\n'
        '}\n\n'
    )
    assert anchor in s
    s = s.replace(anchor, impl + anchor)
    open("/srv/workspace/flychess/src/Stockfish-act/src/engine.cpp", "w").write(s)
    print("engine.cpp ok")
else:
    print("engine.cpp already patched")

# uci.cpp: strip broken inline block, add clean dispatch
s = open("/srv/workspace/flychess/src/Stockfish-act/src/uci.cpp").read()
start = s.find('    else if (token == "rawft")')
if start != -1:
    end = s.find('else if (token == "eval")', start)
    seg = s[start:end]
    s = s.replace(seg, "")
    print("uci.cpp broken block stripped")
if 'token == "rawft"' not in s:
    old = '        else if (token == "eval")\n            engine.trace_eval();'
    new = (
        '        else if (token == "eval")\n'
        '            engine.trace_eval();\n'
        '        else if (token == "rawft")\n'
        '        {\n'
        '            std::string rest;\n'
        '            std::getline(is, rest);\n'
        '            engine.rawft(rest);\n'
        '        }'
    )
    assert old in s, "eval dispatch anchor"
    s = s.replace(old, new)
open("/srv/workspace/flychess/src/Stockfish-act/src/uci.cpp", "w").write(s)
print("uci.cpp ok")
