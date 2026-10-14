#!/usr/bin/env python3
"""Shard the nongame (puzzle) fens for d12 relabeling. 400k fens per shard,
fen<TAB>tag lines — the depth_worker.py input format."""
import os
import sqlite3

con = sqlite3.connect("file:/srv/workspace/chess-active/store/games.db?mode=ro",
                      uri=True)
OUT = "/srv/workspace/chess-active/puzzle-shards"
os.makedirs(OUT, exist_ok=True)

i = 0
f = None
n_sh = 0
for (fen,) in con.execute(
        "select fen from nongame where label glob ?", ("[-0-9]*",)):
    if i % 400000 == 0:
        if f:
            f.close()
        f = open(f"{OUT}/pz.{n_sh:03d}.shard", "w")
        n_sh += 1
    f.write(fen + "\tt\n")
    i += 1
if f:
    f.close()
print(f"shards: {n_sh}  fens: {i:,}")
