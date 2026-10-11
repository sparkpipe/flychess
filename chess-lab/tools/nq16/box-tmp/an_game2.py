import json, sys, urllib.request, re
pair, gi = sys.argv[1], int(sys.argv[2])
pgn = open(f"/mnt/cold-raid6/chess-audit/smoke/{pair}.pgn").read()
games = pgn.split("[Event")[1:]
gtxt = games[gi]
wh = re.search(r"\[White .([^\"]+).", gtxt).group(1)
v5_white = wh.startswith("v5")
d = json.load(urllib.request.urlopen(f"http://localhost:8077/api/analyze?p={pair}&g={gi}", timeout=300))
if "error" in d:
    print("ERROR", d["error"]); sys.exit()
evs, moves = d["evals"], d["moves"]
print(f"game {gi+1}: v5 as {"White" if v5_white else "Black"}, {len(moves)} plies")
worst = []
for k in range(len(moves)):
    b, a = evs[k]["cp"], evs[k+1]["cp"]
    if b is None or a is None: continue
    mover = 1 if k % 2 == 0 else -1
    worst.append((b*mover - a*mover, k, moves[k], evs[k]["best"]))
worst.sort(reverse=True)
for label, is_v5 in [("v5", v5_white), ("nQ", not v5_white)]:
    side_moves = [(L, k, mv, best) for L, k, mv, best in worst if (k % 2 == 0) == is_v5]
    total = sum(x[0] for x in side_moves)
    cnt = len(side_moves)
    bl = [x for x in side_moves if x[0] >= 100]
    print(f"  {label}: avg self-harm {total/max(cnt,1)/100:.2f} pn/move, blunders(>=1pn): {len(bl)}")
    for L, k, mv, best in bl[:3]:
        dots = "..." if k % 2 else "."
        print(f"    {k//2+1}{dots} {mv}  -{L/100:.1f}pn (best: {best})")
n = len(moves)
v5sign = 1 if v5_white else -1
for phase, lo, hi in [("opening", 0, n//3), ("middlegame", n//3, 2*n//3), ("endgame", 2*n//3, n)]:
    vals = [evs[k+1]["cp"] * v5sign for k in range(lo, hi) if evs[k+1]["cp"] is not None]
    if vals:
        print(f"  {phase}: v5-view start {vals[0]:.0f} mid {vals[len(vals)//2]:.0f} end {vals[-1]:.0f} cp")
