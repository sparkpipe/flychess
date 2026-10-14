#!/usr/bin/env python3
"""Fix: append clean short wins to the existing general3 bins."""
import hashlib
import json
import sqlite3
import struct
import sys

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess
from pack_expert_bins import pack_sfen, pack_move

DB = "/srv/workspace/chess-active/store/games.db"
OUT = "/srv/workspace/chess-active/bins-general3"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
clean = json.load(open("/srv/workspace/chess-active/clean_short_wins.json"))

tr = open(f"{OUT}/general3.train.bin", "ab")
va = open(f"{OUT}/general3.val.bin", "ab")
n = 0
for g in clean:
    plies = con.execute(
        "select p.fen, l.cp from plies p"
        " join labels l on p.fen = l.fen"
        "   and l.engine = 'sf17' and l.depth = 20"
        " where p.gid = ? order by p.ply", (g["gid"],)).fetchall()
    for fen, cp in plies:
        try:
            board = chess.Board(fen)
            if not board.is_valid():
                continue
            mv = next(iter(board.legal_moves))
        except Exception:
            continue
        h = hashlib.md5(fen.encode()).hexdigest()
        tag = "val" if int(h, 16) % 20 == 0 else "train"
        fh = va if tag == "val" else tr
        fh.write(struct.pack("<32shHHbB", pack_sfen(board),
                             max(-30000, min(30000, int(cp))),
                             pack_move(mv), board.fullmove_number, 0, 0))
        n += 1
tr.close()
va.close()
print(f"clean short win positions appended: {n:,}")
import os
print(f"train: {os.path.getsize(f'{OUT}/general3.train.bin') // 40:,} recs")
print(f"val: {os.path.getsize(f'{OUT}/general3.val.bin') // 40:,} recs")
