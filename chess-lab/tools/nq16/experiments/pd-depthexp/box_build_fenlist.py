#!/usr/bin/env python3
"""Replicates db_export_v3.py source order/dedup/split EXACTLY, but emits the
piece_down fen list (fen<TAB>train|val) for the eval-depth experiment."""
import hashlib, os, sqlite3, sys
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
from routerB import routeB

DB = "/srv/workspace/chess-active/store/games.db"
OUT = "/srv/workspace/chess-active/pd-depthexp/piece_down.fens"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
seen = set()
n = tr = va = 0
with open(OUT, "w") as f:
    for fen, in con.execute("select fen from seg_positions"):
        if fen in seen: continue
        seen.add(fen)
        if routeB(fen) != "piece_down": continue
        tag = "val" if int(hashlib.md5(fen.encode()).hexdigest(), 16) % 20 == 0 else "train"
        f.write(f"{fen}\t{tag}\n"); tr += tag=="train"; va += tag=="val"; n += 1
        if n % 500000 == 0: print(f"  seg {n:,}", flush=True)
    for fen, label in con.execute("select fen, label from nongame where label glob '[-0-9]*'"):
        if fen in seen: continue
        seen.add(fen)
        if routeB(fen) != "piece_down": continue
        tag = "val" if int(hashlib.md5(fen.encode()).hexdigest(), 16) % 20 == 0 else "train"
        f.write(f"{fen}\t{tag}\n"); tr += tag=="train"; va += tag=="val"; n += 1
        if n % 500000 == 0: print(f"  puzzle {n:,}", flush=True)
    for fen, in con.execute("select p.fen from positions p join labels l on l.fen=p.fen and l.engine='sf17' and l.depth=12 order by p.seq"):
        if fen in seen: continue
        seen.add(fen)
        if routeB(fen) != "piece_down": continue
        tag = "val" if int(hashlib.md5(fen.encode()).hexdigest(), 16) % 20 == 0 else "train"
        f.write(f"{fen}\t{tag}\n"); tr += tag=="train"; va += tag=="val"; n += 1
        if n % 500000 == 0: print(f"  mini {n:,}", flush=True)
print(f"piece_down total={n:,} train={tr:,} val={va:,}")
print(f"EXPECTED train=2,599,209 val=136,595 -> {"MATCH" if (tr,va)==(2599209,136595) else "MISMATCH - DO NOT PROCEED"}")
