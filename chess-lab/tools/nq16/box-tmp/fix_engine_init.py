p = "/srv/workspace/flychess/src/Stockfish-act/src/engine.cpp"
t = open(p).read()
start = t.index("networkFiles{")
end_marker = "}},  // phase-moe"
end = t.index(end_marker, start) + len(end_marker)
unit = 'Eval::NNUE::EvalFile{std::nullopt, ""}'
block = "networkFiles{" + ",\n                 ".join([unit] * 12) + "}},  // phase-moe"
t = t[:start] + block + t[end:]
open(p, "w").write(t)
print("block replaced, units =", block.count(unit))
