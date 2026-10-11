p = "/srv/workspace/flychess/src/Stockfish-act/src/engine.cpp"
t = open(p).read()

ns = t.index("networks{{LazyNumaReplicatedSystemWide<NN::Network>(")
# end: the closing of the networks brace before ",  // phase-moe" or blank line + pos.set
tail = t[ns:]
end_rel = tail.index("\n\n")
seg = tail[:end_rel]
n = seg.count("LazyNumaReplicatedSystemWide<NN::Network>(")
want = 23
if n != want:
    head = seg[:seg.index("),\n              LazyNumaReplicatedSystemWide")] + ")"
    unit = (",\n              LazyNumaReplicatedSystemWide<NN::Network>(\n"
            "                numaContext, std::make_unique<NN::Network>())")
    new_seg = head + unit * (want - 1)
    t = t[:ns] + new_seg + t[ns + end_rel:]
    open(p, "w").write(t)
    print(f"networks: {n} -> {want}")
else:
    print(f"networks already {n}")

# resize_threads
import re
m = re.search(r"\{\{&networks\[0\](?:,\s*&networks\[\d+\])*\}\}", t)
if m:
    cnt = m.group(0).count("&networks[")
    print(f"resize_threads: {cnt} entries")
    if cnt != want:
        new_list = "{{" + ", ".join(f"&networks[{i}]" for i in range(want)) + "}}"
        t = t[:m.start()] + new_list + t[m.end():]
        open(p, "w").write(t)
        print(f"resize_threads -> {want}")
