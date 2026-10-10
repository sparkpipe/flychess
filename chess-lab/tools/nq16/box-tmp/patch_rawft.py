
s = open("/srv/workspace/flychess/src/Stockfish-act/src/uci.cpp").read()
if "rawft" not in s:
    anchor = ''
    # find the eval command handler to mirror
    import re
    m = re.search(r'(\n\s*)else if \(token == "eval"\)', s)
    assert m, "eval command anchor"
    idx = s.index(m.group(0))
    block = """
    else if (token == "rawft")
    // phase-moe: dump raw active feature indices for the current position (stm)
    {
        Position pos;
        StateListPtr states(new std::deque<StateInfo>(1));
        pos.set(UCI::startFen, false, &states->back(), threads.main_thread()->chess960);
        std::string fen;
        std::getline(is, fen);
        if (!fen.empty())
            pos.set(fen, false, &states->back(), threads.main_thread()->chess960);
        const Color us = pos.side_to_move();
        NNUE::ThreatFeatureSet::IndexList tlist;
        NNUE::ThreatFeatureSet::append_active_indices(us, pos, tlist);
        NNUE::PairFeatureSet::IndexList plist;
        NNUE::PairFeatureSet::append_active_indices(us, pos, plist);
        std::string out;
        for (auto i : tlist)
            out += std::to_string(u32(i)) + " ";
        for (auto i : plist)
            out += std::to_string(u32(i)) + " ";
        sync_cout << "rawft " << out << sync_endl;
    }"""
    s = s[:idx] + block + s[idx:]
    open("/srv/workspace/flychess/src/Stockfish-act/src/uci.cpp", "w").write(s)
    print("uci.cpp rawft added")
else:
    print("already present")
