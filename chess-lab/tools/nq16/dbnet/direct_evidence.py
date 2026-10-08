#!/usr/bin/env python3
"""Direct: 5 c-region part-file rows + 5 worker-era DB rows vs fresh SF17 stm-pov."""
import sqlite3
import subprocess

SF = "/srv/workspace/flychess/src/sf17/src/stockfish"
PART = "/srv/workspace/chess-active/miniature_labels17.tsv.part"
DB = "/srv/workspace/chess-active/store/games.db"
K = 7886687

rows = []
with open(PART) as f:
    for i, l in enumerate(f):
        if i < K:
            continue
        if i >= K + 200:
            break
        fen, cp = l.rstrip("\n").split("\t")
        if fen.split()[1] == "b":
            rows.append(("c-part", fen, int(cp)))
        if len(rows) >= 5:
            break

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
for fen, cp in con.execute(
        "select fen, cp from labels where by_node='box0'"
        " and substr(fen, instr(fen,' ')+1, 1)='b' limit 5"):
    rows.append(("worker-db", fen, cp))

p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\nisready\n")
p.stdin.flush()
while "readyok" not in p.stdout.readline():
    pass
for tag, fen, stored in rows:
    p.stdin.write(f"position fen {fen}\ngo depth 12\n")
    p.stdin.flush()
    last = None
    while True:
        l = p.stdout.readline()
        if not l:
            break
        if l.startswith("info ") and " score cp " in l and " pv " in l:
            t = l.split()
            for i, x in enumerate(t):
                if x == "cp":
                    last = int(t[i + 1])
        if l.startswith("bestmove"):
            break
    rel = "AGREE" if (last is not None and stored * last > 0) else "INVERT"
    print(f"{tag}: stored={stored:+6d} fresh_stm={last:+6d} -> {rel}")
p.stdin.write("quit\n")
