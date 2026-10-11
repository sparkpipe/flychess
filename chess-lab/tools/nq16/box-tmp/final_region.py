A = "/srv/workspace/flychess/src/Stockfish/src/engine.cpp"
s = open(A).read()
a = s.index("    threads(),")
b = s.index("pos.set(StartFEN")
one = '                 Eval::NNUE::EvalFile{std::nullopt, ""}'
first = ("    networks{{LazyNumaReplicatedSystemWide<NN::Network>(\n"
         "                numaContext, get_default_network(networkFiles[0])),")
rest_open = "              LazyNumaReplicatedSystemWide<NN::Network>(\n                numaContext, std::make_unique<NN::Network>())"
parts = ["    threads(),", "    networkFiles{" + one.strip() + ","]
parts += [one + ","] * 14
parts.append(one + "}},")
parts.append("")
parts.append(first + ",")
parts += [rest_open + ","] * 14
parts.append(rest_open + "}} {")
parts.append("")
region = "\n".join(parts)
s2 = s[:a] + region + s[b:]
open(A, "w").write(s2)
n16a = s2.count("Eval::NNUE::EvalFile{std::nullopt")
n16b = s2.count("LazyNumaReplicatedSystemWide<NN::Network>(")
print("entries:", n16a, n16b)
