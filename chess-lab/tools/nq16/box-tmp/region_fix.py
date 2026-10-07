A = "/srv/workspace/flychess/src/Stockfish/src/engine.cpp"
lines = open(A).read().split("\n")
# find region: start = line starting with "    networkFiles{", end = line containing "())}},} {"
si = next(i for i, l in enumerate(lines) if l.strip().startswith("networkFiles{"))
ei = next(i for i, l in enumerate(lines) if "())}},} {" in l or (l.rstrip().endswith("}}") and "make_unique" in l))
one = '                 Eval::NNUE::EvalFile{std::nullopt, ""}'
block = ["    networkFiles{" + one[17:] if False else "    networkFiles{" + one[len("    networkFiles{"):] ]
# simpler: build explicit
block = ["    networkFiles{" + one.strip()]
block += [one] * 15
block[-1] = one + "}},"
block.append("")
first = ("    networks{{LazyNumaReplicatedSystemWide<NN::Network>(\n"
         "                numaContext, get_default_network(networkFiles[0])),")
block.append(first)
rest = ("              LazyNumaReplicatedSystemWide<NN::Network>(\n"
        "                numaContext, std::make_unique<NN::Network>())")
block.append(rest + ",")
block += [rest + ","] * 14
block[-1] = rest
block.append("}}")
block.append("")
block.append("    {")
new_lines = lines[:si] + ["\n".join(block)] + lines[ei+1:]
# drop the original body-brace line if the next line is now `pos.set` (it kept its own?) — check
out = "\n".join(new_lines)
open(A, "w").write(out)
print(f"region {si}..{ei} rewritten")
