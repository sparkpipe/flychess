"""Churn comparison: old 13-router (engine ground truth) vs new 23-router
over the SAME sampled segment trajectories.

Feeds position+route to the old engine via stdin batch; route23 in python
for the new router; both count adjacent-ply changes within each segment.
"""
import sys, os, subprocess, random
from collections import defaultdict, Counter

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import router12 as R

SEG = "/extnvme/segments"
ENG = "/srv/workspace/flychess/src/Stockfish/src/stockfish"
random.seed(11)

# collect segment groups (sig=pos only), sample ~120k groups
rows_by_game = defaultdict(list)
files = [f for f in sorted(os.listdir(SEG))
         if f.startswith("pos_") and (f.endswith(".clean.tsv") or f.endswith(".draws.tsv"))]
for fn in files:
    for line in open(f"{SEG}/{fn}", errors="ignore"):
        p = line.rstrip("\n").split("|")
        if len(p) < 6 or p[3] != "pos":
            continue
        gid = p[5]
        if gid.isdigit():
            rows_by_game[(fn, gid)].append(p[0])

keys = list(rows_by_game)
random.shuffle(keys)
keys = keys[:120000]
print(f"sampled {len(keys)} groups", flush=True)

p = subprocess.Popen([ENG], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
def send(x):
    p.stdin.write(x + "\n"); p.stdin.flush()
send("uci")
while "uciok" not in p.stdout.readline():
    pass

ch13 = 0; ch23 = 0; tr = 0
sw13 = Counter(); sw23 = Counter()
BATCH = 256
buf_keys = []
buf_fens = []

def flush_batch():
    global ch13, ch23, tr
    if not buf_fens:
        return
    for fen in buf_fens:
        send("position fen " + fen)
        send("route")
    out = []
    while len(out) < len(buf_fens):
        l = p.stdout.readline()
        if l.startswith("route "):
            out.append(l.split()[1])
    i = 0
    for fens in buf_keys:
        prev13 = prev23 = None
        for fen in fens:
            r13 = out[i]; i += 1
            r23 = R.route23(fen)
            if prev13 is not None:
                tr += 1
                if r13 != prev13:
                    ch13 += 1
                    sw13[f"{prev13}->{r13}"] += 1
                if r23 != prev23:
                    ch23 += 1
                    sw23[f"{prev23}->{r23}"] += 1
            prev13, prev23 = r13, r23

n = 0
for k in keys:
    fens = rows_by_game[k]
    if len(fens) < 2:
        continue
    buf_keys.append(fens)
    buf_fens.extend(fens)
    if len(buf_fens) >= BATCH:
        flush_batch()
        buf_keys, buf_fens = [], []
    n += 1
    if n % 20000 == 0:
        print(f"{n} groups, {tr} transitions, 13churn={100*ch13/max(1,tr):.1f}% 23churn={100*ch23/max(1,tr):.1f}%", flush=True)
flush_batch()
send("quit")

print(f"\nTRANSITIONS: {tr}")
print(f"nQ (13-router) churn: {ch13} = {100*ch13/max(1,tr):.1f}% of adjacent plies")
print(f"nQ.23 (23-router) churn: {ch23} = {100*ch23/max(1,tr):.1f}%")
print("\ntop-13router boundaries:")
for k, v in sw13.most_common(12):
    print(f"  {k}: {v}")
print("\ntop-23router boundaries:")
for k, v in sw23.most_common(12):
    print(f"  {k}: {v}")
