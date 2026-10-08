#!/usr/bin/env python3
"""Locate flipped seed rows by part-file line position (one pass)."""
import random
import sqlite3
import subprocess

random.seed(7)
SF = "/srv/workspace/flychess/src/sf17/src/stockfish"
PART = "/srv/workspace/chess-active/miniature_labels17.tsv.part"
con = sqlite3.connect("file:/srv/workspace/chess-active/store/games.db?mode=ro", uri=True)
rows = con.execute(
    "select fen,cp from labels where engine='sf17' and depth=12 and by_node='box-relabel17'"
    " and substr(fen, instr(fen,' ')+1,1)='b' order by random() limit 40").fetchall()
p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\nisready\n")
p.stdin.flush()
while "readyok" not in p.stdout.readline():
    pass


def agree(fen, stored):
    p.stdin.write(f"position fen {fen}\ngo depth 12\n")
    p.stdin.flush()
    last = None
    while True:
        l = p.stdout.readline()
        if not l:
            return None
        if l.startswith("info ") and " score cp " in l and " pv " in l:
            t = l.split()
            for i, x in enumerate(t):
                if x == "cp":
                    last = int(t[i + 1])
        elif l.startswith("bestmove"):
            break
    return last is not None and stored * last > 0


marks = []
for fen, stored in rows:
    a = agree(fen, stored)
    if a is not None:
        marks.append((fen, a))
p.stdin.write("quit\n")
want = {f for f, a in marks}
pos = {}
with open(PART) as f:
    for i, l in enumerate(f):
        fen = l.split("\t")[0]
        if fen in want:
            pos[fen] = i
            if len(pos) == len(want):
                break
for f, a in sorted(marks, key=lambda x: pos.get(x[0], -1)):
    tag = "OK  " if a else "FLIP"
    print(f"line {pos.get(f, -1):>10,}  {tag}")
