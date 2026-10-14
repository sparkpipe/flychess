#!/usr/bin/env python3
"""Export review games as PGNs — abort on any broken ply (no drift)."""
import json
import os
import sqlite3

import chess
import chess.pgn

DB = "/srv/workspace/chess-active/store/games.db"
OUT = "/srv/workspace/chess-active/matches"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
games = json.load(open("/srv/workspace/chess-active/clean_wins.json"))

# categorize all candidates
cats = {"questionable_highstart": [], "questionable_lowend": [],
        "good_climb": []}
for g in games:
    if g["start_eval"] > 100 and g["end_eval"] < g["start_eval"]:
        cats["questionable_highstart"].append(g)
    elif g["end_eval"] < 60 and g["moves"] >= 7:
        cats["questionable_lowend"].append(g)
    elif g["start_eval"] < 50 and g["end_eval"] > 150 and g["max_jump"] < 100:
        cats["good_climb"].append(g)

print({k: len(v) for k, v in cats.items()})


def try_export(g, label):
    gid = g["gid"]
    meta = con.execute(
        "select white, black, result, welo, belo, date from games where gid=?",
        (gid,)).fetchone()
    if not meta:
        return False
    white, black, result, welo, belo, date = meta
    plies = con.execute(
        "select fen from plies where gid=? order by ply", (gid,)).fetchall()
    if len(plies) < 4:
        return False

    game = chess.pgn.Game()
    start_board = chess.Board(plies[0][0])
    game.setup(start_board)
    game.headers["White"] = white or "?"
    game.headers["Black"] = black or "?"
    game.headers["Result"] = result
    game.headers["WhiteElo"] = str(welo or "")
    game.headers["BlackElo"] = str(belo or "")
    game.headers["Date"] = str(date or "")
    game.headers["Event"] = f"review_{label}"

    node = game
    prev = chess.Board(plies[0][0])
    for i in range(1, len(plies)):
        nxt = chess.Board(plies[i][0])
        mv = None
        for candidate in list(prev.legal_moves):
            test = prev.copy()
            test.push(candidate)
            if test.board_fen() == nxt.board_fen() and test.turn == nxt.turn:
                mv = candidate
                break
        if mv is None:
            return False  # abort — no drift allowed
        node = node.add_variation(mv)
        prev = nxt

    fname = f"review_{label}_gid{gid}"
    with open(os.path.join(OUT, fname + ".pgn"), "w") as f:
        f.write(str(game))
        f.write("\n\n")
    print(f"  {fname}: {len(plies)-1} plies, {result}, "
          f"start={g['start_eval']} end={g['end_eval']} jump={g['max_jump']}")
    return True


exported = 0
for label, candidates in cats.items():
    for g in candidates[:10]:  # try up to 10 per category
        if try_export(g, label):
            exported += 1
            if exported >= 6:
                break
    if exported >= 6:
        break
print(f"total exported: {exported}")
