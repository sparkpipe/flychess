"""Label miniature rows with SF d12 evals (white-pov cp), 6 workers, resumable."""
import subprocess, threading, os, re

SF = "/usr/games/stockfish"
SRC = "/extnvme/active/miniature_rows.tsv"
OUT = "/extnvme/active/miniature_labels.tsv"
TMPOUT = OUT + ".tmp"
N = 6

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

procs = [P() for _ in range(N)]
done = {}
if os.path.exists(OUT):
    for line in open(OUT):
        h, cp = line.rstrip("\n").split("\t")
        done[h] = cp
out = open(TMPOUT, "w")
n = 0
buf = []
def flushbuf():
    global n
    for fen in buf:
        cp = procs[n % N].eval1(fen)
        if cp is not None:
            out.write(f"{fen}\t{cp}\n")
        n += 1
        if n % 50000 == 0:
            print(f"{n:,} labeled", flush=True)
    buf.clear()

for line in open(SRC):
    fen = line.split("\t")[0]
    if fen in done:
        continue
    buf.append(fen)
    if len(buf) >= 2000:
        flushbuf()
flushbuf()
out.close()
os.replace(TMPOUT, OUT)
print(f"DONE {n:,} newly labeled ({len(done):,} already present)")
