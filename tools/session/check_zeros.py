import json

d = json.load(open("/srv/workspace/chess-active/matches/games.json"))
g = d[2]["games"][0]
evs = g.get("evals_d20", [])
fens = g["fens"]
zeros = [i for i, v in enumerate(evs) if v == 0]
print(f"d20 len={len(evs)} plies={len(fens)} zeros={len(zeros)}")
if zeros:
    lo, hi = min(zeros), max(zeros)
    print(f"zero range: ply {lo}..{hi}")
    print(f"fen@{lo}: {fens[lo]}")
    print(f"evals around: {evs[max(0,lo-2):hi+3]}")
