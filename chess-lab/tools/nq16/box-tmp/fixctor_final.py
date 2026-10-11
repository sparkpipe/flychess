p = "/srv/workspace/flychesis/src/Stockfish-act/src/engine.cpp"
p = "/srv/workspace/flychess/src/Stockfish-act/src/engine.cpp"
t = open(p).read()

# Rebuild the whole constructor head cleanly from the git version's structure:
# member init list: ..., networkFiles{...}, networks{ single brace, 23 entries, no inner double brace }
i = t.index("networkFiles{")
j = t.index("LazyNumaReplicatedSystemWide", t.index("networks{{"))
# capture from networkFiles to just after the last entry's close
k = t.index("pos.set(StartFEN")
head = t[:i]
tail = t[k:]

unit_f = 'Eval::NNUE::EvalFile{std::nullopt, ""}'
files_block = "networkFiles{" + ",\n                 ".join([unit_f] * 23) + "}},\n\n"

net_head = ("networks{LazyNumaReplicatedSystemWide<NN::Network>(\n"
            "                numaContext, get_default_network(networkFiles[0]))")
net_unit = (",\n              LazyNumaReplicatedSystemWide<NN::Network>(\n"
            "                numaContext, std::make_unique<NN::Network>())")
nets_block = net_head + net_unit * 22 + "}},\n\n    {\n        "

t = head + files_block + nets_block + tail
open(p, "w").write(t)
print("constructor head rebuilt cleanly")
