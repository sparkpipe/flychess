import json, sys, urllib.request, re
pair, g = sys.argv[1], sys.argv[2]
d = json.load(urllib.request.urlopen(f"http://localhost:8077/api/analyze?p={pair}&g={g}", timeout=300))
if "error" in d: print(pair, g, "ERROR", d["error"]); sys.exit()
evs, moves = d["evals"], d["moves"]
worst = []
for k in range(len(moves)):
    b, a = evs[k]["cp"], evs[k+1]["cp"]
    if b is None or a is None: continue
    mover = 1 if k % 2 == 0 else -1
    worst.append((b*mover - a*mover, k, moves[k], evs[k]["best"]))
worst.sort(reverse=True)
num = 1 if "g0" not in g else int(g[1:])
# print self-harm profile for the v5 side (white in odd games of v5fair)
pgn = open(f"/mnt/cold-raid6/chess-audit/smoke/{pair}.pgn").read()
games = pgn.split("[Event")[1:]
gtxt = games[num]
wh = re.search(r"\[White .([^\"]+).", gtxt).group(1)
v5_white = wh.startswith("v5")
print(f"game {num}: v5 as {chr(87) if v5_white else chr(66)}")
# per-side self-harm profile
for label, is_v5 in [("v5", v5_white), ("nQ", not v5_white)]:
    blunders = [(L, k, mv, best) for L, k, mv, best in worst
                if (k % 2 == 0) == v5_white and L >= 100]
    total = sum(L for L, k, mv, best in worst if (k % 2 == 0) == v5_white)
    cnt = sum(1 for L, k, mv, best in worst if (k % 2 == 0) == v5_white)
    print(f"  {label}: avg self-harm {total/max(cnt,1)/100:.2f} pawns/move, blunders(>=1pn)={len(blunders)}")
    for L, k, mv, best in blunders[:3]:
        side = "W" if k % 2 == 0 else "B"
        print(f"    mv{k//2+1}{'...' if k%2 else '.'} {mv} ({side}) -{L/100:.1f}pn best={best}")
# trajectory: early/mid/late eval for v5 side
n = len(moves)
for phase, lo, hi in [("opening", 0, n//3), ("middlegame", n//3, 2*n//3), ("endgame", 2*n//3, n)]:
    vals = [evs[k+1]["cp"] for k in range(lo, hi) if evs[k+1]["cp"] is not None]
    if vals:
        v5sign = 1 if v5_white else -1
        v5ev = [v * v5sign for v in vals]
        print(f"  {phase}: v5 eval mean {sum(v5ev)/len(v5ev):.0f} start {v5ev[0]:.0f} end {v5ev[-1]:.0f}")
