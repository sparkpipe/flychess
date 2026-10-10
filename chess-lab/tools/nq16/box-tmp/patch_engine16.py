import re

A = "/srv/workspace/flychess/src/Stockfish-act/src"
open(f"{A}/phase_moe.h.bak23", "w").write(open(f"{A}/phase_moe.h").read())

s = open(f"{A}/engine.cpp").read()
s = re.sub(r"networkFiles\{[^}]*\}\},", "networkFiles{" + ", ".join(["{}"] * 16) + "}},", s, count=1)
s = re.sub(r"networks\{\{[^}]*\}\}", "networks{{" + ", ".join(["{}"] * 16) + "}}", s, count=1)
open(f"{A}/engine.cpp", "w").write(s)

u = open(f"{A}/uci.cpp").read()
names = ["tb", "mvr", "rv2m", "qvmat", "nvb", "piece_down", "oppb", "dv_Q",
         "dv_R", "dv_rest", "op_pawnimb", "op_even_l0", "op_even_l1", "op_even_l2p",
         "mg_unsafe", "mg_safe"]
rows = ",\n".join('              "%s"' % n for n in names)
new_block = "            static const char* PhaseMoeNames[] = {\n" + rows + "};"
u2 = re.sub(r"            static const char\* PhaseMoeNames\[\] = \{[^;]*\};", new_block, u, count=1, flags=re.S)
assert u2 != u, "names block not found"
open(f"{A}/uci.cpp", "w").write(u2)
print("engine.cpp + uci.cpp updated for 16 slots")
