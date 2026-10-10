A = "/srv/workspace/flychess/src/Stockfish/src/engine.cpp"
s = open(A).read()

def replace_braced(s, start_marker, new_block):
    i = s.index(start_marker)
    j = i + len(start_marker) - 1   # at the '{'
    depth = 0
    k = j
    while k < len(s):
        if s[k] == '{':
            depth += 1
        elif s[k] == '}':
            depth -= 1
            if depth == 0:
                end = k + 1
                # consume trailing chars up to and including ',' if present
                while end < len(s) and s[end] in ' ,\t':
                    if s[end] == ',':
                        end += 1
                        break
                    end += 1
                return s[:i] + new_block + s[end:]
        k += 1
    raise SystemExit("unbalanced")

one = 'Eval::NNUE::EvalFile{std::nullopt, ""}'
nf = "networkFiles{" + ",\n                 ".join([one] * 16) + "}},"
s = replace_braced(s, "networkFiles{", nf)

first = ("LazyNumaReplicatedSystemWide<NN::Network>(\n"
         "                numaContext, get_default_network(networkFiles[0]))")
rest = ("LazyNumaReplicatedSystemWide<NN::Network>(\n"
        "                numaContext, std::make_unique<NN::Network>())")
nw = "networks{{" + first + ",\n              " + ",\n              ".join([rest] * 15) + "}},"
s = replace_braced(s, "networks{{", nw)
open(A, "w").write(s)
print("brace-aware 16-slot constructor written")
