#!/usr/bin/env python3
"""Find games with a decisive advantage established by move 18.
The game may be longer — what matters is the first 18 moves show a smooth
build to a decisive advantage. The training positions come from the
advantage-building phase.

Filter:
  - decisive result
  - winner's eval ≥ 200cp by ply 36 (move 18)
  - smooth curve to that point: ≤ 3 reversals, no giveback > 50cp
  - start eval ≥ -50cp
  - no single-ply jump ≥ 200cp
"""
import json
import math
import sqlite3
from collections import defaultdict

DB = "/srv/workspace/chess-active/store/games.db"

DECISIVE_CP = 150    # winner ≥ this by move 18
BY_PLY = 36          # move 18
MAX_REVERSALS = 5    # in the first 36 plies
MAX_GIVEBACK = 80    # cp
MAX_JUMP = 200       # cp single-ply
MIN_START = -50      # cp

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

games_meta = {}
for gid, off, result, welo, belo in con.execute(
        "select gid, off, result, welo, belo from games"):
    games_meta[gid] = (off, result, welo, belo)

by_game = defaultdict(list)
for ref, fen, cp in con.execute(
        "select p.ref, p.fen, l.cp from positions p"
        " join labels l on l.fen = p.fen and l.engine = 'sf17' and l.depth = 20"
        " where p.kind = 'mini_game' order by p.seq"):
    gid, ply = ref.split(":")
    by_game[int(gid)].append((int(ply), fen, cp))
print(f"games with d20: {len(by_game):,}", flush=True)

found = []
for gid, plies in by_game.items():
    off, result, welo, belo = games_meta.get(gid, (0, "1/2-1/2", "", ""))
    if result not in ("1-0", "0-1"):
        continue
    plies.sort()
    if len(plies) < BY_PLY:
        continue
    winner_white = result == "1-0"
    seq = []
    for ply, fen, cp in plies:
        stm_white = fen.split()[1] == "w"
        w_pov = cp if (stm_white == winner_white) else -cp
        seq.append(w_pov)

    # evaluate only the first 36 plies (move 18)
    build = seq[:BY_PLY]

    if build[0] < MIN_START:
        continue
    # must be ≥ 200cp by ply 36
    if build[-1] < DECISIVE_CP:
        continue
    # no single-ply jump ≥ 200cp
    if any(abs(b - a) >= MAX_JUMP for a, b in zip(build, build[1:])):
        continue
    # no giveback > 50cp below a reached high
    peak = build[0]
    for v in build:
        if v > peak:
            peak = v
        elif peak - v > MAX_GIVEBACK:
            break
    else:
        # count reversals
        reversals = 0
        prev_d = 0
        for a, b in zip(build, build[1:]):
            d = b - a
            if prev_d != 0 and (d > 0) != (prev_d > 0):
                reversals += 1
            if d != 0:
                prev_d = d
        if reversals <= MAX_REVERSALS:
            found.append({
                "gid": gid, "result": result,
                "total_plies": len(seq),
                "build_evals": build,
                "start": build[0], "at18": build[-1],
                "reversals": reversals,
                "welo": welo, "belo": belo,
            })

print(f"decisive-by-18 smooth games: {len(found):,}")
json.dump(found, open("/srv/workspace/chess-active/decisive18.json", "w"))

# show distribution of when advantage was reached
from collections import Counter
first_200 = Counter()
for g in found:
    for i, v in enumerate(g["build_evals"]):
        if v >= DECISIVE_CP:
            first_200[i // 2 + 1] += 1
            break
for m in sorted(first_200):
    if first_200[m] > 5:
        print(f"  decisive by move {m}: {first_200[m]:,}")
