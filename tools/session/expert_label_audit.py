#!/usr/bin/env python3
"""Print raw label-vs-fresh values for the failing experts' mini and otb
rows — are failures TT-warmth tails or sign-flipped/corrupt subsets?"""
import sqlite3
import subprocess
import sys

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess
sys.path.insert(0, "/srv/workspace/chess-active")
from debug_gate_dec import unpack_sfen  # standalone decoder module
import struct

DB = "/srv/workspace/chess-active/store/games.db"
SF = "/srv/workspace/flychess/src/sf17/src/stockfish"
BINS = "/srv/workspace/chess-active/bins-correct16"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\nsetoption name Threads value 1\nisready\n")
p.stdin.flush()
while "readyok" not in p.stdout.readline():
    pass


def d12(fen):
    p.stdin.write(f"ucinewgame\nposition fen {fen}\ngo depth 12\n")
    p.stdin.flush()
    cp = None
    while True:
        ln = p.stdout.readline()
        if not ln:
            return None
        if ln.startswith("info") and not ln.startswith("info string") \
                and " score " in ln:
            t = ln.split()
            for i, x in enumerate(t):
                if i and t[i-1] == "score" and x in ("cp", "mate"):
                    try:
                        v = int(t[i+1])
                    except (ValueError, IndexError):
                        break
                    cp = ((10000 + min(abs(v), 900)) * (1 if v > 0 else -1)
                          if x == "mate" else v)
                    break
        elif ln.startswith("bestmove"):
            return cp


for expert in ("mg_unsafe", "op_even_l0", "op_pawnimb", "tb", "dv_Q"):
    raw = open(f"{BINS}/{expert}.train.bin", "rb").read()
    n = len(raw) // 40
    import random
    random.seed(hash(expert) % 1000)
    shown = 0
    print(f"\n=== {expert} ===")
    for i in random.sample(range(n), min(400, n)):
        sfen, score, mv, ply, b, u = struct.unpack_from("<32shHHbB", raw, i * 40)
        fen = unpack_sfen(sfen)
        if not fen:
            continue
        # classify
        fk = tuple(fen.split()[:3])
        cls = None
        if con.execute("select 1 from seg_positions where fen like ?"
                       " limit 1", (fen.split()[0] + " " + fen.split()[1] + " %",)).fetchone():
            cls = "otb"
        elif con.execute("select 1 from nongame where fen like ?"
                         " limit 1", (fen.split()[0] + " " + fen.split()[1] + " %",)).fetchone():
            cls = "puz"
        else:
            cls = "mini"
        fresh = d12(fen)
        if fresh is None:
            continue
        d = abs(score - fresh)
        if d > 50 or score * fresh < 0:
            shown += 1
            print(f"  [{cls}] bin={score:>6} fresh={fresh:>6} d={d:>5}"
                  f" flip={score * fresh < 0}  {fen[:44]}")
            if shown >= 8:
                break
    if shown == 0:
        print("  (no mismatches >50cp or sign-flips in sample)")
