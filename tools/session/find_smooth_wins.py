#!/usr/bin/env python3
"""Updated clean-win filter: adds smoothness (no big giveback once ahead).
Once the winner's eval reaches a new high, it can't drop more than 75cp
below that high. Also limits direction reversals.

Exports qualifying games as one PGN group for UI review."""
import json
import math
import os
import sqlite3
from collections import defaultdict

import chess
import chess.pgn

DB = "/srv/workspace/chess-active/store/games.db"
OUT = "/srv/workspace/chess-active/matches"

MAX_MOVES = 18
MAX_JUMP = 250
MIN_LATE_EVAL = 100
MIN_START_EVAL = -50
LATE_PLIES = 8
MAX_GIVEBACK = 75   # cp, can't drop this far below a reached high
MAX_REVERSALS = 6   # eval direction changes

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

smooth_wins = []
for gid, plies in by_game.items():
    off, result, welo, belo = games_meta.get(gid, (0, "1/2-1/2", "", ""))
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
        seq.append(w_pov)

    if seq[0] < MIN_START_EVAL:
        continue
    if any(abs(b - a) >= MAX_JUMP for a, b in zip(seq, seq[1:])):
        continue
    if len(seq) <= LATE_PLIES:
        continue
    if seq[-(LATE_PLIES + 1)] < MIN_LATE_EVAL:
        continue

    # NEW: smoothness — no big giveback below a reached high
    peak = seq[0]
    smooth = True
    for v in seq:
        if v > peak:
            peak = v
        elif peak - v > MAX_GIVEBACK:
            smooth = False
            break
    if not smooth:
        continue

    # NEW: limit direction reversals
    reversals = 0
    prev_delta = 0
    for a, b in zip(seq, seq[1:]):
        delta = b - a
        if prev_delta != 0 and (delta > 0) != (prev_delta > 0):
            reversals += 1
        if delta != 0:
            prev_delta = delta
    if reversals > MAX_REVERSALS:
        continue

    smooth_wins.append({
        "gid": gid, "result": result,
        "moves": len(seq) // 2, "plies": len(seq),
        "curve": seq,
        "start": seq[0], "end": seq[-1],
        "max_jump": max(abs(b - a) for a, b in zip(seq, seq[1:])),
        "reversals": reversals,
    })

print(f"smooth wins: {len(smooth_wins):,}")
from collections import Counter
for m, c in sorted(Counter(g["moves"] for g in smooth_wins).items()):
    print(f"  {m} moves: {c:,}")

# export 6 as one PGN group (diverse move counts)
json.dump(smooth_wins, open("/srv/workspace/chess-active/smooth_wins.json", "w"))
selected = []
by_len = defaultdict(list)
for g in smooth_wins:
    by_len[g["moves"]].append(g)
for m in sorted(by_len):
    selected.extend(by_len[m][:2])
selected = selected[:6]

with open(os.path.join(OUT, "review_smooth_wins.pgn"), "w") as out:
    for g in selected:
        gid = g["gid"]
        meta = con.execute(
            "select white, black, result, welo, belo, date from games where gid=?",
            (gid,)).fetchone()
        if not meta:
            continue
        white, black, result, welo, belo, date = meta
        plies = con.execute(
            "select fen from plies where gid=? order by ply", (gid,)).fetchall()
        if len(plies) < 4:
            continue
        game = chess.pgn.Game()
        game.setup(chess.Board(plies[0][0]))
        game.headers["White"] = white or "?"
        game.headers["Black"] = black or "?"
        game.headers["Result"] = result
        game.headers["Event"] = "review_smooth_wins"
        node = game
        prev = chess.Board(plies[0][0])
        ok = True
        for i in range(1, len(plies)):
            nxt = chess.Board(plies[i][0])
            mv = None
            for c in list(prev.legal_moves):
                t = prev.copy()
                t.push(c)
                if t.board_fen() == nxt.board_fen() and t.turn == nxt.turn:
                    mv = c
                    break
            if mv is None:
                ok = False
                break
            node = node.add_variation(mv)
            prev = nxt
        if ok:
            out.write(str(game))
            out.write("\n\n")
            print(f"  exported {white} vs {black}: {g['moves']} moves, "
                  f"start={g['start']} end={g['end']} rev={g['reversals']}")
print("DONE")
