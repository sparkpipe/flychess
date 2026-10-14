#!/usr/bin/env python3
"""Ground-truth check: sidecar d12 cp+pv vs a fresh clean SF17 d12 run."""
import glob
import json
import subprocess

samples = []
for sc in glob.glob("/Users/mac/games-live/sidecars/*.evals.json"):
    side = json.load(open(sc))
    for fen, e in side.items():
        if "12" in e and "12pv" in e and "|" in e["12pv"]:
            samples.append((fen, e["12"], e["12pv"]))
        if len(samples) >= 3:
            break
    if len(samples) >= 3:
        break

p = subprocess.Popen(["/Users/mac/sf-local/sf17/src/stockfish"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\nsetoption name Threads value 1\nisready\n")
p.stdin.flush()
while "readyok" not in p.stdout.readline():
    pass

for fen, cp, pvline in samples:
    p.stdin.write(f"ucinewgame\nposition fen {fen}\ngo depth 12\n")
    p.stdin.flush()
    gcp, gpv = None, ""
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
                    gcp = raw * (1 if fen.split()[1] == "w" else -1)
                    break
            if " pv " in ln:
                gpv = ln.split(" pv ", 1)[1].strip()
        elif ln.startswith("bestmove"):
            break
    stored_cp, stored_pv = pvline.split("|", 1)
    print(f"{fen[:44]}")
    print(f"  stored cp={stored_cp} pv={stored_pv[:46]}")
    print(f"  truth  cp={gcp} pv={gpv[:46]}"
          f"  cp_match={int(stored_cp) == gcp}"
          f" pv_match={stored_pv.split()[:6] == gpv.split()[:6]}")
p.stdin.write("quit\n")
p.stdin.flush()
