#!/usr/bin/env python3
"""What's the winner's max eval in the miniature corpus? How many games
actually reach a decisive (rook-level, 500cp+) advantage?"""
import sqlite3
from collections import Counter, defaultdict

con = sqlite3.connect(
    "file:/srv/workspace/chess-active/store/games.db?mode=ro", uri=True)
games_meta = {}
for gid, result in con.execute("select gid, result from games"):
    games_meta[gid] = result

by_game = defaultdict(list)
for ref, fen, cp in con.execute(
        "select p.ref, p.fen, l.cp from positions p"
        " join labels l on l.fen = p.fen and l.engine = 'sf17' and l.depth = 20"
        " where p.kind = 'mini_game' order by p.seq"):
    gid, ply = ref.split(":")
    by_game[int(gid)].append((int(ply), fen, cp))

max_evals = Counter()
for gid, plies in by_game.items():
    result = games_meta.get(gid, "1/2-1/2")
    if result not in ("1-0", "0-1"):
        continue
    plies.sort()
    winner_white = result == "1-0"
    best = -99999
    for ply, fen, cp in plies:
        stm_white = fen.split()[1] == "w"
        w_pov = cp if (stm_white == winner_white) else -cp
        if w_pov > best:
            best = w_pov
    bucket = "never>100" if best < 100 else \
             "100-300" if best < 300 else \
             "300-500" if best < 500 else \
             "500-800" if best < 800 else "800+"
    max_evals[bucket] += 1

total = sum(max_evals.values())
print(f"decisive miniature games: {total:,}")
print("winner max eval distribution:")
for b in ["never>100", "100-300", "300-500", "500-800", "800+"]:
    n = max_evals.get(b, 0)
    print(f"  {b}: {n:,} ({100*n/total:.1f}%)")
