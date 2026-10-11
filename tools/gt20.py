"""Ground truth: depth-20 (12s cap) evals of test positions, system SF 17.1."""
import os, subprocess, sys
IN, OUT = os.environ["INPUT"], os.environ["OUTPUT"]
WID, NW = int(os.environ.get("WORKER_ID", "0")), int(os.environ.get("NUM_WORKERS", "20"))
SF = "/usr/games/stockfish"
with open(IN) as f:
    lines = f.readlines()
total = len(lines)
start, end = WID * total // NW, (WID + 1) * total // NW
done = 0
wpath = OUT + f".w{WID}"
if os.path.exists(wpath):
    done = sum(1 for _ in open(wpath))
pos = start + done
if pos >= end:
    sys.exit(0)
def spawn():
    p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\n")
    while not p.stdout.readline().startswith("uciok"):
        pass
    p.stdin.write("setoption name Threads value 1\nsetoption name Hash value 128\nisready\n")
    while not p.stdout.readline().startswith("readyok"):
        pass
    return p
p = spawn()
n = 0
with open(IN) as fin, open(wpath, "a") as fout:
    for i, line in enumerate(fin):
        if i < pos:
            continue
        if i >= end:
            break
        fen = line.split("|")[0]
        try:
            p.stdin.write("position fen %s\ngo depth 20 movetime 12000\n" % fen)
            p.stdin.flush()
            cp = None
            while True:
                r = p.stdout.readline()
                if not r:
                    raise RuntimeError("died")
                if r.startswith("info ") and " score cp " in r:
                    toks = r.split()
                    cp = int(toks[toks.index("cp") + 1])
                elif r.startswith("bestmove"):
                    break
            if cp is None:
                continue
            fout.write("%s|%d\n" % (line.rstrip("\n"), cp))
            n += 1
            if n % 500 == 0:
                fout.flush()
                print("w%d: %d/%d" % (WID, n, end - pos), flush=True)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
            p = spawn()
try:
    p.stdin.write("quit\n")
except Exception:
    pass
