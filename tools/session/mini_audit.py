#!/usr/bin/env python3
"""Why do some mini d12 labels mismatch fresh evals? Sample mini fens from
the failing experts, compare labels-d12 vs fresh (stm-pov), print pattern."""
import sqlite3
import subprocess

DB = "/srv/workspace/chess-active/store/games.db"
SF = "/srv/workspace/flychess/src/sf17/src/stockfish"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rows = con.execute(
    "select p.fen, l.cp from positions p join labels l"
    " on l.fen=p.fen and l.engine='sf17' and l.depth=12"
    " order by random() limit 40").fetchall()

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


exact = near = flip = other = 0
for fen, lab in rows:
    fresh = d12(fen)
    if fresh == lab:
        exact += 1
    elif fresh is not None and abs(fresh - lab) <= 30:
        near += 1
    elif fresh is not None and fresh == -lab:
        flip += 1
        print(f"FLIP: labels={lab} fresh={fresh}  {fen[:50]}")
    else:
        other += 1
        print(f"DIFF: labels={lab} fresh={fresh}  {fen[:50]}")
print(f"\n40 random mini d12: exact={exact} near={near} SIGN-FLIP={flip}"
      f" other={other}")
