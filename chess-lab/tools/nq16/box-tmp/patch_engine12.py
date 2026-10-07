p = "/srv/workspace/flychess/src/Stockfish-act/src/engine.cpp"
t = open(p).read()

# 1. networkFiles: 13 -> 12 (remove exactly one unit line)
unit = '                 Eval::NNUE::EvalFile{std::nullopt, ""},\n'
assert t.count(unit) == 11, t.count(unit)
i = t.rindex(unit)
t = t[:i] + t[i + len(unit):]

# 2. networks: 13 entries -> 12 (remove one make_unique entry, keeping the default)
net_unit = ("              LazyNumaReplicatedSystemWide<NN::Network>(\n"
            "                numaContext, std::make_unique<NN::Network>()),\n")
assert t.count(net_unit) == 11, t.count(net_unit)
i = t.rindex(net_unit)
t = t[:i] + t[i + len(net_unit):]

# 3. resize_threads list: drop &networks[12]
old3 = "                   &networks[11],\n                   &networks[12]}},"
assert old3 in t
t = t.replace(old3, "                   &networks[11]}},")
open(p, "w").write(t)
print("engine.cpp patched: 12 slots")
