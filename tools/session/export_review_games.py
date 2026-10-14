#!/usr/bin/env python3
"""Export questionable + good clean-win games as PGNs for UI review.
Reconstructs the game by walking fens through python-chess."""
import json
import os
import sqlite3

import chess
import chess.pgn

DB = "/srv/workspace/chess-active/store/games.db"
OUT = "/srv/workspace/chess-active/matches"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
games = json.load(open("/srv/workspace/chess-active/clean_wins.json"))

picks = []
for g in games:
    if g["start_eval"] > 100 and g["end_eval"] < g["start_eval"]:
        picks.append(("questionable_highstart", g))
for g in games:
    if g["end_eval"] < 60 and g["moves"] >= 7:
        picks.append(("questionable_lowend", g))
for g in games:
    if g["start_eval"] < 50 and g["end_eval"] > 150 and g["max_jump"] < 100:
        picks.append(("good_climb", g))

seen = set()
selected = []
for label, g in picks:
    if g["gid"] not in seen:
        n = len([p for p in selected if p[0].startswith(label.split("_")[0])])
        if n < 2:
            seen.add(g["gid"])
            selected.append((label, g))

print(f"exporting {len(selected)} games")
for label, g in selected:
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
    game.headers["White"] = white or "?"
    game.headers["Black"] = black or "?"
    game.headers["Result"] = result
    game.headers["WhiteElo"] = str(welo or "")
    game.headers["BlackElo"] = str(belo or "")
    game.headers["Date"] = str(date or "")
    game.headers["Event"] = f"review_{label}"

    node = game
    prev = chess.Board(plies[0][0])
    ok = True
    for i in range(1, len(plies)):
        nxt = chess.Board(plies[i][0])
        mv = None
        for candidate in prev.legal_moves:
            test = prev.copy()
            test.push(candidate)
            if test.fen().split()[0] == nxt.fen().split()[0]:
                mv = candidate
                break
        if mv is None:
            ok = False
            break
        node = node.add_variation(mv)
        prev = nxt
    if not ok:
        continue
    fname = f"review_{label}_gid{gid}"
    with open(os.path.join(OUT, fname + ".pgn"), "w") as f:
        f.write(str(game))
        f.write("\n\n")
    print(f"  {fname}: {g['moves']} moves, {result}, "
          f"start={g['start_eval']} end={g['end_eval']} jump={g['max_jump']}")
