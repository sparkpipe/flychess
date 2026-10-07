import re
A = "/srv/workspace/flychess/src/Stockfish/src"
# 1) phase_moe.h: 16-slot version
open(f"{A}/phase_moe.h.bak13", "w").write(open(f"{A}/phase_moe.h").read())
new = open("/tmp/phase_moe16.h").read()
# adapt: the 16-header uses PhaseMoESlots already; fine as-is
open(f"{A}/phase_moe.h", "w").write(new)

# 2) engine.cpp: count entries
s = open(f"{A}/engine.cpp").read()
m = re.search(r"networkFiles\{(.*?)\}\},", s, flags=re.S)
n_entries = m.group(1).count("EvalFile{")
print("networkFiles entries:", n_entries)
one = 'Eval::NNUE::EvalFile{std::nullopt, ""}'
s = s[:m.start()] + "networkFiles{" + ",\n                 ".join([one] * 16) + "}}," + s[m.end():]
m2 = re.search(r"networks\{\{(.*?)\}\}", s, flags=re.S)
print("networks entries:", m2.group(1).count("LazyNumaReplicatedSystemWide"))
first = ("LazyNumaReplicatedSystemWide<NN::Network>(\n"
         "                numaContext, get_default_network(networkFiles[0]))")
rest = ("LazyNumaReplicatedSystemWide<NN::Network>(\n"
        "                numaContext, std::make_unique<NN::Network>())")
s = s[:m2.start()] + "networks{{" + first + ",\n              " + \
    ",\n              ".join([rest] * 15) + "}}" + s[m2.end():]
open(f"{A}/engine.cpp", "w").write(s)

# 3) uci.cpp names
u = open(f"{A}/uci.cpp").read()
names = ["tb", "mvr", "rv2m", "qvmat", "nvb", "piece_down", "oppb", "dv_Q",
         "dv_R", "dv_rest", "op_pawnimb", "op_even_l0", "op_even_l1", "op_even_l2p",
         "mg_unsafe", "mg_safe"]
rows = ",\n".join('              "%s"' % n for n in names)
block = "            static const char* PhaseMoeNames[] = {\n" + rows + "};"
u2, cnt = re.subn(r"            static const char\* PhaseMoeNames\[\] = \{[^;]*\};", block, u, count=1, flags=re.S)
print("uci names replaced:", cnt)
open(f"{A}/uci.cpp", "w").write(u2)
print("PORT COMPLETE")
