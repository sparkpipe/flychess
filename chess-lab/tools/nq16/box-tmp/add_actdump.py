s = open("/srv/workspace/flychess/src/Stockfish-act/src/engine.h").read()
if "actdump" not in s:
    s = s.replace("    void rawft(const std::string& fen) const;",
                  "    void rawft(const std::string& fen) const;\n    void actdump(const std::string& fen) const;  // phase-moe: dump slot0 stm activated block")
    open("/srv/workspace/flychess/src/Stockfish-act/src/engine.h", "w").write(s)
    print("engine.h ok")

s = open("/srv/workspace/flychess/src/Stockfish-act/src/engine.cpp").read()
if "Engine::actdump" not in s:
    anchor = "const OptionsMap& Engine::get_options() const { return options; }"
    impl = (
        'void Engine::actdump(const std::string& fen) const {\n'
        '    StateListPtr states(new std::deque<StateInfo>(1));\n'
        '    Position     p;\n'
        '    std::string  f = fen;\n'
        '    while (!f.empty() && f.front() == \' \')\n'
        '        f.erase(f.begin());\n'
        '    p.set(f.empty() ? pos.fen() : f, options["UCI_Chess960"], &states->back());\n'
        '    alignas(64) i8 acts[1024];\n'
        '    networks[0]->evaluate_capture(p, accumulatorStacks[0], refreshTables[0], acts);\n'
        '    std::string out;\n'
        '    for (int i = 0; i < 1024; i++)\n'
        '        out += std::to_string(int(acts[i])) + " ";\n'
        '    sync_cout << "actdump " << out << sync_endl;\n'
        '}\n\n'
    )
    assert anchor in s
    s = s.replace(anchor, impl + anchor)
    open("/srv/workspace/flychess/src/Stockfish-act/src/engine.cpp", "w").write(s)
    print("engine.cpp ok")

s = open("/srv/workspace/flychess/src/Stockfish-act/src/uci.cpp").read()
if 'token == "actdump"' not in s:
    old = '        else if (token == "rawft")'
    new = (
        '        else if (token == "actdump")\n'
        '        {\n'
        '            std::string rest;\n'
        '            std::getline(is, rest);\n'
        '            engine.actdump(rest);\n'
        '        }\n'
        '        else if (token == "rawft")'
    )
    assert old in s
    s = s.replace(old, new, 1)
    open("/srv/workspace/flychess/src/Stockfish-act/src/uci.cpp", "w").write(s)
    print("uci.cpp ok")
