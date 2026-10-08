#!/usr/bin/env python3
"""FINAL accuracy audit: random sample across ALL labeled sources vs fresh SF17 d12.
PASS bar: stm-sign >= 95%, median |err| sane, scale ~1.0."""
import math
import random
import sqlite3
import subprocess

random.seed(42)
DB = "/srv/workspace/chess-active/store/games.db"
SF17 = "/srv/workspace/flychess/src/sf17/src/stockfish"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rows = con.execute(
    "select p.fen, l.cp from positions p join labels l "
    "on l.fen=p.fen and l.engine='sf17' and l.depth=12 order by random() limit 300").fetchall()

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
                elif t[i - 1] == "score" and x == "mate":
                    v = int(t[i + 1])
                    last = (10000 + min(abs(v), 900)) * (1 if v > 0 else -1)
        elif l.startswith("bestmove"):
            return last


n = stm_ok = 0
errs = []
xs, ys = [], []
for fen, lab in rows:
    t = cp(fen)
    if t is None:
        continue
    n += 1
    if lab * t > 0 or (abs(lab) < 50 and abs(t) < 50):
        stm_ok += 1
    if abs(lab) < 9000 and abs(t) < 9000:
        errs.append(abs(lab - t))
        xs.append(t)
        ys.append(lab)
p.stdin.write("quit\n")
errs.sort()
mx = sum(xs) / len(xs)
my = sum(ys) / len(ys)
cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / len(xs)
vx = sum((a - mx) ** 2 for a in xs) / len(xs)
vy = sum((b - my) ** 2 for b in ys) / len(ys)
corr = cov / math.sqrt(vx * vy)
scale = cov / vx
print(f"FINAL AUDIT: n={n}")
print(f"  sign/agree: {stm_ok}/{n} = {100*stm_ok/n:.1f}%")
print(f"  corr (stored vs fresh SF17): {corr:+.4f}")
print(f"  scale (OLS stored/fresh): {scale:.3f}")
print(f"  median |err|: {errs[len(errs)//2]:.0f} cp   p90: {errs[int(len(errs)*0.9)]:.0f} cp")
ok = stm_ok / n >= 0.95 and 0.9 <= scale <= 1.1 and corr > 0.95
print("VERDICT:", "PASS" if ok else "FAIL")
