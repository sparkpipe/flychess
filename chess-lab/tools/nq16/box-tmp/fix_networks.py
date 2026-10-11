A = "/srv/workspace/flychess/src/Stockfish-act/src"
lines = open(f"{A}/engine.cpp").read().split("\n")
# find the networks line (idx)
for i, l in enumerate(lines):
    if l.strip().startswith("networks{{") and "LazyNuma" not in l:
        head = "    networks{{LazyNumaReplicatedSystemWide<Eval::NNUE::Network>("
        block = [head + "numaContext, get_default_network(networkFiles[0])),"]
        for _ in range(15):
            block.append("              LazyNumaReplicatedSystemWide<Eval::NNUE::Network>("
                         "numaContext, std::make_unique<Eval::NNUE::Network>())")
        block[-1] += "}"
        block.append("")
        lines[i:i+1] = ["\n".join(block).rstrip("}").rstrip() + "}}"]
        break
else:
    raise SystemExit("networks line not found")
open(f"{A}/engine.cpp", "w").write("\n".join(lines))
print("networks init rewritten to constructor form")
