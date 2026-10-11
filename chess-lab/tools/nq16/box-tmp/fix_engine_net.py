p = "/srv/workspace/flychess/src/Stockfish-act/src/engine.cpp"
t = open(p).read()
start = t.index("networks{{")
end_marker = "}},  // phase-moe"
end = t.index(end_marker, start) + len(end_marker)
head = ("networks{{LazyNumaReplicatedSystemWide<NN::Network>(\n"
        "                numaContext, get_default_network(networkFiles[0]))")
unit = ("LazyNumaReplicatedSystemWide<NN::Network>(\n"
        "                numaContext, std::make_unique<NN::Network>())")
block = head + ",\n              " + ",\n              ".join([unit] * 11) + "}},  // phase-moe"
t = t[:start] + block + t[end:]
open(p, "w").write(t)
print("networks block replaced, total nets =", block.count("LazyNumaReplicatedSystemWide<NN::Network>("))
