import re
A = "/srv/workspace/flychess/src/Stockfish-act/src"
s = open(f"{A}/engine.cpp").read()
# replace the whole region from networkFiles{ to the networks initializer end
m = re.search(r"networkFiles\{.*?networks\{\{.*?\}\}", s, flags=re.S)
assert m, "region not found"
one = 'Eval::NNUE::EvalFile{std::nullopt, ""}'
block = ("networkFiles{" + ",\n                 ".join([one] * 16) + "}},\n\n"
         + "    networks{{" + ", ".join(["{}"] * 16) + "}}")
s = s[:m.start()] + block + s[m.end():]
open(f"{A}/engine.cpp", "w").write(s)
print("clean 16-slot constructor written")
