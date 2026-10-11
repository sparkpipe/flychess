import re, sys
P = sys.argv[1]
txt = open(P, errors="ignore").read()
games = re.findall(r"\[Event.*?(?=\[Event|\Z)", txt, re.S)
score = 0.0
for i, g in enumerate(games):
    w = re.search(r"\[White \"(\w+)\"\]", g)
    b = re.search(r"\[Black \"(\w+)\"\]", g)
    r = re.search(r"\[Result \"([\d-]+)\"\]", g).group(1)
    plies = re.search(r"\[PlyCount \"(\d+)\"\]", g)
    me = 1.0 if r == "1-0" else 0.0 if r == "0-1" else 0.5
    color = "W"
    if b and b.group(1) == "nQ23":
        color = "B"; me = 1.0 - me
    res = "win" if me == 1 else "loss" if me == 0 else "draw"
    print(f"game {i+1}: nQ23={color} {r} plies={plies.group(1) if plies else '?'} -> {res}")
    score += me
print(f"nQ23 score: {score}/{len(games)}  ({100*score/max(1,len(games)):.0f}%)")
