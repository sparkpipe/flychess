s = open("/srv/workspace/flychess/src/Stockfish-act/src/uci.cpp").read()
broken_start = '    else if (token == "rawft")\n    // phase-moe: dump raw active feature indices for the current position (stm)\n    {'
assert broken_start in s, "broken block start not found"
i = s.index(broken_start)
j = s.index('        else if (token == "eval")', i)
clean = (
    '        else if (token == "rawft")\n'
    '        {\n'
    '            std::string rest;\n'
    '            std::getline(is, rest);\n'
    '            engine.rawft(rest);\n'
    '        }\n'
)
s = s[:i] + clean + s[j:]
assert s.count('token == "rawft"') == 1
open("/srv/workspace/flychess/src/Stockfish-act/src/uci.cpp", "w").write(s)
print("uci.cpp cleaned")
