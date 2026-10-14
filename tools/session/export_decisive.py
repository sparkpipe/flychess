#!/usr/bin/env python3
"""Export 6 diverse decisive-by-18 games as one PGN group for UI review."""
import json
import os
import sqlite3
from collections import defaultdict

import chess
import chess.pgn

DB = "/srv/workspace/chess-active/store/games.db"
OUT = "/srv/workspace/chess-active/matches"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
games = json.load(open("/srv/workspace/chess-active/decisive18.json"))

# pick 6 diverse: 2 from move 3-4, 2 from move 5-6, 2 from move 7-9
selected = []
for lo, hi in [(3, 4), (5, 6), (7, 9)]:
    bucket = [g for g in games if lo <= g["decisive_at_move"] <= hi]
    # pick with lowest reversal counts (smoothest)
    bucket.sort(key=lambda g: g["reversals"])
    selected.extend(bucket[:2])

print(f"exporting {len(selected)} games")
with open(os.path.join(OUT, "review_decisive18.pgn"), "w") as out:
    for g in selected:
        gid = g["gid"]
        meta = con.execute(
            "select white, black, result, welo, belo from games where gid=?",
            (gid,)).fetchone()
        if not meta:
            continue
        white, black, result, welo, belo = meta
        plies = con.execute(
            "select fen from plies where gid=? order by ply", (gid,)).fetchall()
        if len(plies) < 4:
            continue
        game = chess.pgn.Game()
        game.setup(chess.Board(plies[0][0]))
        game.headers["White"] = white or "?"
        game.headers["Black"] = black or "?"
        game.headers["Result"] = result
        game.headers["Event"] = "review_decisive18"
        node = game
        prev = chess.Board(plies[0][0])
        ok = True
        # export only up to the decisive point + 4 more plies
        limit = min(len(plies), g["decisive_at_ply"] + 5)
        for i in range(1, limit):
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
            print(f"  {white} vs {black}: decisive at move {g['decisive_at_move']}, "
                  f"start={g['start']} → {g['at_decisive']}cp, "
                  f"rev={g['reversals']}, total_plies={g['total_plies']}")
print("DONE")
