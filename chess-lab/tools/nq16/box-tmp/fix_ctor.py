import re
A = "/srv/workspace/flychess/src/Stockfish-act/src"
s = open(f"{A}/engine.cpp").read()
one = 'Eval::NNUE::EvalFile{std::nullopt, ""}'
m = re.search(r"networkFiles\{.*?\}\},", s, flags=re.S)
assert m, "networkFiles block not found"
s = s[:m.start()] + "networkFiles{" + ",\n                 ".join([one] * 16) + "}}," + s[m.end():]
m2 = re.search(r"networks\{\{.*?\}\}", s, flags=re.S)
assert m2, "networks block not found"
s = s[:m2.start()] + "networks{{" + ", ".join(["{}"] * 16) + "}}" + s[m2.end():]
open(f"{A}/engine.cpp", "w").write(s)
print("constructor now 16/16")
