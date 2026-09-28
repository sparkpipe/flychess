"""Raw-UCI depth-12 eval worker — no python-chess engine wrapper (its event
loop died repeatedly under load), resumable by existing output count.

Usage env: INPUT, OUTPUT, WORKER_ID, NUM_WORKERS, DEPTH, SF
Input lines:  fen|move|tag|ply|elo1|elo2|gid
Output lines: fen|move|tag|ply|elo1|elo2|gid|cp||best|depth
"""
import os
import subprocess
import sys
import time

IN = os.environ["INPUT"]
OUT = os.environ["OUTPUT"]
WID = int(os.environ.get("WORKER_ID", "0"))
NW = int(os.environ.get("NUM_WORKERS", "8"))
DEPTH = os.environ.get("DEPTH", "12")
SF = os.environ.get("SF", "/home/spec/Stockfish/src/stockfish")

with open(IN) as f:
    total = sum(1 for _ in f)
start = WID * total // NW
end = (WID + 1) * total // NW if WID < NW - 1 else total

done = 0
if os.path.exists(OUT):
    with open(OUT) as f:
        done = sum(1 for _ in f)
pos = start + done
if pos >= end:
    print("worker %d: already complete" % WID)
    sys.exit(0)

p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\n")
while True:
    line = p.stdout.readline()
    if line.startswith("uciok"):
        break
p.stdin.write("setoption name Threads value 1\nsetoption name Hash value 128\nisready\n")
while True:
    line = p.stdout.readline()
    if line.startswith("readyok"):
        break

t0 = time.time()
n = fails = 0
with open(IN) as fin, open(OUT, "a") as fout:
    for i, line in enumerate(fin):
        if i < pos:
            continue
        if i >= end:
            break
        parts = line.rstrip("\n").split("|")
        if len(parts) < 7:
            continue
        fen = parts[0]
        try:
            p.stdin.write("position fen %s\ngo depth %s\n" % (fen, DEPTH))
            p.stdin.flush()
            cp = best = None
            while True:
                r = p.stdout.readline()
                if not r:
                    raise RuntimeError("engine died")
                if r.startswith("info ") and " score cp " in r:
                    toks = r.split()
                    cp = int(toks[toks.index("cp") + 1])
                    try:
                        pv = toks.index("pv")
                        best = toks[pv + 1]
                    except ValueError:
                        pass
                elif r.startswith("bestmove"):
                    if best is None:
                        best = r.split()[1]
                    break
            if cp is None:
                cp = 0
            fout.write("%s|%s\n" % ("|".join(parts[:7]), cp))
            fout.write("")  # no flush per line; buffered
            n += 1
        except Exception:
            fails += 1
            # restart engine
            try:
                p.kill()
            except Exception:
                pass
            p = subprocess.Popen([SF], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True, bufsize=1)
            p.stdin.write("uci\n")
            while True:
                if p.stdout.readline().startswith("uciok"):
                    break
            p.stdin.write("setoption name Threads value 1\n"
                          "setoption name Hash value 128\nisready\n")
            while True:
                if p.stdout.readline().startswith("readyok"):
                    break
        if n % 10000 == 0 and n:
            fout.flush()
            print("worker %d: %d/%d (%d fails, %.0f/s)"
                  % (WID, n, end - pos, fails, n / (time.time() - t0)),
                  flush=True)

fout = None
try:
    p.stdin.write("quit\n")
    p.wait(timeout=5)
except Exception:
    p.kill()
print("worker %d DONE: %d evaluations, %d fails, %ds"
      % (WID, n, fails, time.time() - t0))
