import re
A = "/srv/workspace/flychess/src/Stockfish/src/uci.cpp"
u = open(A).read()
names = ["tb", "mvr", "rv2m", "qvmat", "nvb", "piece_down", "oppb", "dv_Q",
         "dv_R", "dv_rest", "op_pawnimb", "op_even_l0", "op_even_l1", "op_even_l2p",
         "mg_unsafe", "mg_safe"]
rows = ",\n".join('              "%s"' % n for n in names)
block = "            static const char* PhaseMoeNames[] = {\n" + rows + "};"
u2, cnt = re.subn(r"            static const char\* PhaseMoeNames\[\] = \{[^;]*\};", block, u, count=1, flags=re.S)
assert cnt == 1, cnt
open(A, "w").write(u2)
print("16 names in uci.cpp")
