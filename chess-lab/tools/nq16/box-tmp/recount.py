p = "/srv/workspace/flychess/src/Stockfish-act/src/engine.cpp"
t = open(p).read()

unit = 'Eval::NNUE::EvalFile{std::nullopt, ""},\n                 '
start = t.index("networkFiles{")
end = t.index("}},  // phase-moe", start) + len("}},  // phase-moe")
count = t[start:end].count('EvalFile{')
want = 23
if count != want:
    new_block = "networkFiles{" + unit * (want - 1) + 'Eval::NNUE::EvalFile{std::nullopt, ""}},  // phase-moe'
    t = t[:start] + new_block + t[end:]
    open(p, "w").write(t)
    print(f"networkFiles: {count} -> {want}")
else:
    print(f"networkFiles already {count}")

# networks block: count entries
ns = t.index("networks{{")
ne = t.index("}},  // phase-moe", ns) + len("}},  // phase-moe")
seg = t[ns:ne]
n_entries = seg.count("LazyNumaReplicatedSystemWide<NN::Network>(")
want_n = 23
if n_entries != want_n:
    head = seg[:seg.index(",\n")]  # first entry with default network
    unit2 = (",\n              LazyNumaReplicatedSystemWide<NN::Network>(\n"
             "                numaContext, std::make_unique<NN::Network>())")
    new_seg = head + unit2 * (want_n - 1) + "}},  // phase-moe"
    t = t[:ns] + new_seg + t[ne:]
    open(p, "w").write(t)
    print(f"networks: {n_entries} -> {want_n}")
else:
    print(f"networks already {n_entries}")

# resize_threads list
if "&networks[12]}}" in t:
    t = t.replace("                   &networks[11],\n                   &networks[12]}}",
                  "                   &networks[11]}}")
    open(p, "w").write(t)
    print("resize_threads trimmed (was 13-entry)")
elif "&networks[22]}" in t:
    print("resize_threads: verify count manually")
else:
    m = t.find("&networks[")
    print("resize_threads state unclear — inspect")
