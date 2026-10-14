#!/usr/bin/env python3
"""Export questionable + good clean-win games as PGNs for UI review.
Robust fen-walking: full fen comparison with castling/ep, skip broken games."""
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
        n = len([p for p in selected if p[0].split("_")[0] == label.split("_")[0]])
        if n < 2:
            seen.add(g["gid"])
            selected.append((label, g))

print(f"exporting {len(selected)} games")
exported = 0
for label, g in selected:
    gid = g["gid"]
    meta = con.execute(
        "select white, black, result, welo, belo, date from games where gid=?",
        (gid,)).fetchone()
    if not meta:
        continue
    white, black, result, welo, belo, date = meta
    plies = con.execute(
        "select ply, fen from plies where gid=? order by ply", (gid,)).fetchall()
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
    prev = chess.Board(plies[0][1])
    move_count = 0
    for i in range(1, len(plies)):
        nxt_fen = plies[i][1]
        # parse the next fen's board state (placement + stm + castling + ep)
        parts = nxt_fen.split()
        nxt = chess.Board(parts[0] + " " + parts[1] + " " + parts[2] + " " + parts[3])
        mv = None
        for candidate in list(prev.legal_moves):
            test = prev.copy()
            try:
                test.push(candidate)
            except Exception:
                continue
            if test.board_fen() == nxt.board_fen() and test.turn == nxt.turn:
                mv = candidate
                break
        if mv is None:
            continue  # skip broken ply rather than abort
        node = node.add_variation(mv)
        prev = chess.Board(nxt_fen)
        move_count += 1

    if move_count < 4:
        continue
    fname = f"review_{label}_gid{gid}"
    with open(os.path.join(OUT, fname + ".pgn"), "w") as f:
        f.write(str(game))
        f.write("\n\n")
    print(f"  {fname}: {move_count} moves, {result}, "
          f"start={g['start_eval']} end={g['end_eval']} jump={g['max_jump']}")
    exported += 1
print(f"exported: {exported}")
