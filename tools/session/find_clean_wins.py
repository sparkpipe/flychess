#!/usr/bin/env python3
"""Find short decisive games where the win came from sustained pressure,
not a sudden blunder. Uses d20 eval trajectories.

Filter:
  - game ≤ 18 moves (36 plies), decisive result
  - winner's eval at game start ≥ -50cp (half a pawn down is OK)
  - no single-ply jump ≥ 250cp in the winner's trajectory (no blunder-win)
  - winner was already ≥ +100cp at least 8 plies before the end

Output: JSON with game metadata + eval curve for manual review."""
import json
import math
import sqlite3
import sys
from collections import defaultdict

DB = "/srv/workspace/chess-active/store/games.db"
MAX_MOVES = 18
MAX_JUMP = 250       # cp, single-ply
MIN_LATE_EVAL = 100  # cp, this far ahead with 8+ plies to go
MIN_START_EVAL = -50 # cp, can start half a pawn down
LATE_PLIES = 8

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

# load games
games = {}
for gid, off, result, welo, belo in con.execute(
        "select gid, off, result, welo, belo from games"):
    games[gid] = (off, result, welo, belo)

# load d20-labeled positions grouped by game
by_game = defaultdict(list)
rows = con.execute(
    "select p.ref, p.fen, l.cp from positions p"
    " join labels l on l.fen = p.fen and l.engine = 'sf17' and l.depth = 20"
    " where p.kind = 'mini_game'"
    " order by p.seq").fetchall()
for ref, fen, cp in rows:
    gid, ply = ref.split(":")
    by_game[int(gid)].append((int(ply), fen, cp))
print(f"games with d20 trajectories: {len(by_game):,}", file=sys.stderr)

found = []
for gid, plies in by_game.items():
    off, result, welo, belo = games.get(gid, (0, "1/2-1/2", "", ""))
    if result not in ("1-0", "0-1"):
        continue
    plies.sort()
    if len(plies) > MAX_MOVES * 2:
        continue
    winner_white = result == "1-0"
    seq = []
    for ply, fen, cp in plies:
        stm_white = fen.split()[1] == "w"
        w_pov = cp if (stm_white == winner_white) else -cp
        seq.append((ply, w_pov))

    # filter 1: initial eval ≥ -50cp
    if seq[0][1] < MIN_START_EVAL:
        continue
    # filter 2: no single-ply jump ≥ 250cp
    if any(abs(b[1] - a[1]) >= MAX_JUMP for a, b in zip(seq, seq[1:])):
        continue
    # filter 3: winner ≥ +100cp with ≥ 8 plies remaining
    if len(seq) <= LATE_PLIES:
        continue
    late_start = seq[-(LATE_PLIES + 1)]
    if late_start[1] < MIN_LATE_EVAL:
        continue

    found.append({
        "gid": gid,
        "result": result,
        "moves": len(seq) // 2,
        "plies": len(seq),
        "welo": welo, "belo": belo,
        "eval_curve": [c for _, c in seq],
        "max_jump": max(abs(b[1] - a[1]) for a, b in zip(seq, seq[1:])) if len(seq) > 1 else 0,
        "start_eval": seq[0][1],
        "end_eval": seq[-1][1],
    })

print(f"qualifying games: {len(found):,}", file=sys.stderr)
json.dump(found, open("/srv/workspace/chess-active/clean_wins.json", "w"))
# summary histogram
from collections import Counter
move_counts = Counter(g["moves"] for g in found)
for m in sorted(move_counts):
    print(f"  {m} moves: {move_counts[m]:,}")
