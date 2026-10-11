"""Spark-side miniature labeler: sf17_arm d12, W parallel SF procs, resumable."""
import subprocess, sys, threading, os, re

CHUNK = sys.argv[1]
OUT = CHUNK + ".out"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SF = os.path.expanduser("~/sf17_arm")

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        done.add(line.split("\t", 1)[0])

class P:
    def __init__(self):
        self.p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("uci")
        while "uciok" not in self.p.stdout.readline():
            pass
        self.send("setoption name Threads value 1")
        self.send("isready")
        while "readyok" not in self.p.stdout.readline():
            pass
    def send(self, x):
        self.p.stdin.write(x + "\n"); self.p.stdin.flush()
    def eval1(self, fen):
        self.send("position fen " + fen)
        self.send("go depth 12")
        last = ""
        while True:
            l = self.p.stdout.readline()
            if not l:
                return None
            if l.startswith("info") and " score " in l and " pv " in l:
                last = l
            if l.startswith("bestmove"):
                break
        m = re.search(r"score (cp|mate) (-?\d+)", last) if last else None
        if not m:
            return None
        v = int(m.group(2))
        if m.group(1) == "mate":
            v = (10000 + min(abs(v), 900)) if v > 0 else -(10000 + min(abs(v), 900))
        white = fen.split()[1] == "w"
        return v if white else -v

procs = [P() for _ in range(W)]
lock = threading.Lock()
outf = open(OUT, "a", buffering=1)
n = [0]

fens = []
for line in open(CHUNK):
    fen = line.split("\t")[0]
    if fen not in done:
        fens.append(fen)
total = len(fens)
print(f"{CHUNK}: {total:,} to label ({len(done):,} done)", flush=True)

def work(w):
    while True:
        with lock:
            if not fens:
                return
            fen = fens.pop()
        cp = procs[w].eval1(fen)
        if cp is not None:
            outf.write(f"{fen}\t{cp}\n")
        with lock:
            n[0] += 1
            if n[0] % 20000 == 0:
                print(f"{CHUNK}: {n[0]:,}/{total:,}", flush=True)

ts = [threading.Thread(target=work, args=(w,)) for w in range(W)]
for t in ts:
    t.start()
for t in ts:
    t.join()
outf.close()
print(f"{CHUNK}: COMPLETE {n[0]:,}", flush=True)
