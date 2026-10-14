#!/usr/bin/env python3
"""Clean short wins — operator spec, verbatim:
  - winner reaches 2 pawns (200cp) or more
  - once at 2 pawns, NEVER goes below 2 pawns
  - no single blunder jump (≥300cp) before reaching 2 pawns
  - game ends OR winner has rook advantage (500cp) by move 20
  - ENTIRE game used for training (all positions)
"""
import json
import sqlite3
from collections import defaultdict

DB = "/srv/workspace/chess-active/store/games.db"
TWO_PAWNS = 200
ROOK = 500
BY_PLY = 40        # move 20
MAX_JUMP = 300     # no single blunder creates the advantage

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

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
print(f"games with d20: {len(by_game):,}", flush=True)

found = []
for gid, plies in by_game.items():
    result = games_meta.get(gid, "1/2-1/2")
    if result not in ("1-0", "0-1"):
        continue
    plies.sort()
    winner_white = result == "1-0"
    seq = []
    for ply, fen, cp in plies:
        stm_white = fen.split()[1] == "w"
        w_pov = cp if (stm_white == winner_white) else -cp
        seq.append(w_pov)

    # find first ply reaching 2 pawns
    two_pawn_ply = None
    for i, v in enumerate(seq):
        if v >= TWO_PAWNS:
            two_pawn_ply = i
            break
    if two_pawn_ply is None:
        continue

    # no single blunder jump before reaching 2 pawns
    pre = seq[:two_pawn_ply + 1]
    if any(abs(b - a) >= MAX_JUMP for a, b in zip(pre, pre[1:])):
        continue

    # once at 2 pawns, never below
    post = seq[two_pawn_ply:]
    if any(v < TWO_PAWNS for v in post):
        continue

    # game ends OR rook up by move 20
    rook_ply = None
    for i, v in enumerate(seq):
        if v >= ROOK:
            rook_ply = i
            break
    if rook_ply is None:
        continue  # never reached rook
    if rook_ply > BY_PLY:
        continue  # rook advantage came too late

    found.append({
        "gid": gid,
        "result": result,
        "total_plies": len(seq),
        "two_pawn_at_move": two_pawn_ply // 2 + 1,
        "rook_at_move": rook_ply // 2 + 1,
        "start_eval": seq[0],
        "peak_eval": max(seq),
        "build_evals": seq[:rook_ply + 1],
    })

print(f"clean short wins (rook by move 20, no giveback below 2 pawns): "
      f"{len(found):,}")
json.dump(found, open("/srv/workspace/chess-active/clean_short_wins.json", "w"))
from collections import Counter
for m, c in sorted(Counter(g["rook_at_move"] for g in found).items()):
    print(f"  rook by move {m}: {c:,}")
