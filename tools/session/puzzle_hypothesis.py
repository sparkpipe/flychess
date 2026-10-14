#!/usr/bin/env python3
"""What is the nongame puzzle label? For sample fens compare:
  stored label  vs  (a) SF17-d12 eval of the fen itself
                  (b) SF17-d12 eval AFTER the engine's best move
                  (c) mate/scale patterns"""
import sqlite3
import subprocess

con = sqlite3.connect("file:/srv/workspace/chess-active/store/games.db?mode=ro",
                      uri=True)
rows = con.execute(
    "select fen, label from nongame where label glob '[-0-9]*'"
    " order by random() limit 8").fetchall()

p = subprocess.Popen(["/srv/workspace/flychess/src/sf17/src/stockfish"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\nsetoption name Threads value 1\nisready\n")
p.stdin.flush()
while "readyok" not in p.stdout.readline():
    pass


def ev(fen):
    p.stdin.write(f"ucinewgame\nposition fen {fen}\ngo depth 12\n")
    p.stdin.flush()
    cp = None
    best = None
    while True:
        ln = p.stdout.readline()
        if ln.startswith("info") and " score " in ln \
                and not ln.startswith("info string"):
            t = ln.split()
            for i, x in enumerate(t):
                if i and t[i-1] == "score" and x in ("cp", "mate"):
                    v = int(t[i+1])
                    raw = ((10000 + min(abs(v), 900)) * (1 if v > 0 else -1)
                           if x == "mate" else v)
                    cp = raw * (1 if fen.split()[1] == "w" else -1)
                    break
            if " pv " in ln:
                best = ln.split(" pv ", 1)[1].split()[0]
        elif ln.startswith("bestmove"):
            return cp, best


import chess

print(f"{'stored':>7} {'fen_ev':>7} {'after_mv':>8}  bestmove   fen")
for fen, lab in rows:
    cp_now, best = ev(fen)
    cp_after = None
    if best:
        b = chess.Board(fen)
        try:
            b.push(chess.Move.from_uci(best))
            cp_after, _ = ev(b.fen())
        except Exception:
            pass
    print(f"{lab:>7} {cp_now:>7} {str(cp_after):>8}  {str(best):>8}"
          f"   {fen[:44]}")
