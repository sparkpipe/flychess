#!/usr/bin/env python3
"""Verify the era boundary before the targeted flip.
b-era = part-file lines [0, K) with K=7,886,687 (relabel17c's resume point): stm-pov, correct.
c-era = lines [K, end): white-pov flip on black rows.
Samples 60 black-stm fens from each side; checks stored vs fresh SF17 sign.
Only if b-era agrees and c-era inverts do we proceed with the targeted UPDATE."""
import sqlite3
import subprocess

K = 7886687
PART = "/srv/workspace/chess-active/miniature_labels17.tsv.part"
DB = "/srv/workspace/chess-active/store/games.db"
SF17 = "/srv/workspace/flychess/src/sf17/src/stockfish"


def black_lines(lo, hi, n):
    out = []
    with open(PART) as f:
        for i, line in enumerate(f):
            if i < lo:
                continue
            if i >= hi:
                break
            fen, cp = line.rstrip("\n").split("\t")
            if fen.split()[1] == "b":
                out.append((fen, int(cp)))
                if len(out) >= n:
                    break
    return out


con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
p = subprocess.Popen([SF17], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\nisready\n")
p.stdin.flush()
while "readyok" not in p.stdout.readline():
    pass


def cp(fen):
    p.stdin.write(f"position fen {fen}\ngo depth 12\n")
    p.stdin.flush()
    last = None
    while True:
        l = p.stdout.readline()
        if not l:
            return None
        if l.startswith("info ") and " score " in l and " pv " in l:
            t = l.split()
            for i, x in enumerate(t):
                if t[i - 1] == "score" and x == "cp":
                    last = int(t[i + 1])
        elif l.startswith("bestmove"):
            return last


for name, lo, hi in (("b-era", max(0, K - 300000), K),
                     ("c-era", K, K + 300000)):
    agree = flip = n = 0
    for fen, stored in black_lines(lo, hi, 60):
        stored_db = con.execute(
            "select cp from labels where fen=? and engine='sf17' and depth=12",
            (fen,)).fetchone()
        if stored_db is None:
            continue
        s = stored_db[0]
        t = cp(fen)
        if t is None or abs(t) < 30:
            continue
        n += 1
        if s * t > 0:
            agree += 1
        else:
            flip += 1
    print(f"{name}: n={n} agree={agree} inverted={flip}"
          f" -> {'CORRECT (stm-pov)' if agree > flip else 'FLIPPED (white-pov)'}")
p.stdin.write("quit\n")
