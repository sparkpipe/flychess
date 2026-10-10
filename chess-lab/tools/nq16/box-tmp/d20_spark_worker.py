"""Spark d20 worker: reads ~/d20job.txt lines, evaluates sf17_arm at depth 20,
appends fen<TAB>cp to ~/d20out.tsv (resume by counting). 16 SF procs."""
import subprocess, threading, os, re, sys

SF = os.path.expanduser("~/sf17_arm")
W = 16
JOB = os.path.expanduser("~/d20job.txt")
OUT = os.path.expanduser("~/d20out.tsv")

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
        self.send("go depth 20")
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
        return v  # white-pov handled by consumer; store raw stm cp with fen

procs = [P() for _ in range(W)]
ndone = 0
if os.path.exists(OUT):
    ndone = sum(1 for _ in open(OUT))
fens = []
for i, line in enumerate(open(JOB)):
    if i >= ndone:
        fens.append(line.strip())
print(f"{ndone:,} already done, {len(fens):,} to go", flush=True)

lock = threading.Lock()
outf = open(OUT, "a", buffering=1)
n = [0]

def work(w):
    while True:
        with lock:
            if not fens:
                return
            fen = fens.pop()
        cp = procs[w].eval1(fen)
        with lock:
            if cp is not None:
                outf.write(f"{fen}\t{cp}\n")
            n[0] += 1
            if n[0] % 2000 == 0:
                print(f"{n[0]:,}/{len(fens)+n[0]:,}", flush=True)

ts = [threading.Thread(target=work, args=(w,)) for w in range(W)]
for t in ts:
    t.start()
for t in ts:
    t.join()
outf.close()
print("D20 WORKER COMPLETE", flush=True)
