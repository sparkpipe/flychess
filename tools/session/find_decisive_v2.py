#!/usr/bin/env python3
"""Find games where a decisive advantage is REACHED by move 18 (not the
game lasting 18 moves — the advantage is established within 18 moves).
The path to that advantage must be smooth.

Filter:
  - decisive result
  - winner's eval first reaches ≥150cp at or before ply 36 (move 18)
  - smooth path to that point: ≤5 reversals, giveback ≤80cp, jump ≤200cp
  - start eval ≥ -50cp
"""
import json
import math
import sqlite3
from collections import defaultdict

DB = "/srv/workspace/chess-active/store/games.db"
DECISIVE_CP = 150
BY_PLY = 36
MAX_REVERSALS = 5
MAX_GIVEBACK = 80
MAX_JUMP = 200
MIN_START = -50

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

games_meta = {}
for gid, off, result, welo, belo in con.execute(
        "select gid, off, result, welo, belo from games"):
    games_meta[gid] = (off, result, welo, belo)

by_game = defaultdict(list)
for ref, fen, cp in con.execute(
        "select p.ref, p.fen, l.cp from positions p"
        " join labels l on l.fen = p.fen and l.engine = 'sf17' and l.depth = 20"
        " where p.kind = 'mini_game' order by p.seq",
        ):
    gid, ply = ref.split(":")
    by_game[int(gid)].append((int(ply), fen, cp))
print(f"games with d20: {len(by_game):,}", flush=True)

found = []
for gid, plies in by_game.items():
    off, result, welo, belo = games_meta.get(gid, (0, "1/2-1/2", "", ""))
    if result not in ("1-0", "0-1"):
        continue
    plies.sort()
    winner_white = result == "1-0"
    seq = []
    for ply, fen, cp in plies:
        stm_white = fen.split()[1] == "w"
        w_pov = cp if (stm_white == winner_white) else -cp
        seq.append(w_pov)

    if not seq or seq[0] < MIN_START:
        continue
    # advantage must be BUILT, not inherited — start below 100cp
    if seq[0] >= 100:
        continue

    # find first ply where eval ≥ DECISIVE_CP
    decisive_ply = None
    for i, v in enumerate(seq):
        if v >= DECISIVE_CP:
            decisive_ply = i
            break
    if decisive_ply is None or decisive_ply > BY_PLY:
        continue

    # check smoothness only up to the decisive ply
    build = seq[:decisive_ply + 1]
    if len(build) < 2:
        continue
    if any(abs(b - a) >= MAX_JUMP for a, b in zip(build, build[1:])):
        continue
    peak = build[0]
    smooth = True
    for v in build:
        if v > peak:
            peak = v
        elif peak - v > MAX_GIVEBACK:
            smooth = False
            break
    if not smooth:
        continue
    reversals = 0
    prev_d = 0
    for a, b in zip(build, build[1:]):
        d = b - a
        if prev_d != 0 and (d > 0) != (prev_d > 0):
            reversals += 1
        if d != 0:
            prev_d = d
    if reversals > MAX_REVERSALS:
        continue

    found.append({
        "gid": gid, "result": result,
        "total_plies": len(seq),
        "decisive_at_ply": decisive_ply,
        "decisive_at_move": decisive_ply // 2 + 1,
        "build_evals": build,
        "start": build[0],
        "at_decisive": build[-1],
        "reversals": reversals,
        "welo": welo, "belo": belo,
    })

print(f"decisive-by-18 smooth games: {len(found):,}")
json.dump(found, open("/srv/workspace/chess-active/decisive18.json", "w"))
from collections import Counter
for m, c in sorted(Counter(g["decisive_at_move"] for g in found).items()):
    if c > 3:
        print(f"  decisive at move {m}: {c:,}")
