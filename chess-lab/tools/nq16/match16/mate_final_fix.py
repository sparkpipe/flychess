#!/usr/bin/env python3
"""One-shot: final ply of each decided game = ±10000 (checkmate), not 0/empty.
Deterministic from the game result; extends arrays to full length."""
import fcntl
import json
import os

OUT = "/srv/workspace/chess-active/matches/games.json"
DEPTHS = [12, 20, 25]

lock = open(OUT + ".lock", "w")
fcntl.flock(lock, fcntl.LOCK_EX)
doc = json.load(open(OUT))
fixed = 0
for m in doc:
    for g in m["games"]:
        n = len(g["fens"])
        # decisive game whose last move carries the mate symbol
        if g["result"] not in ("1-0", "0-1") or not (g["sans"] and g["sans"][-1].endswith("#")):
            continue
        final = 10000 if g["result"] == "1-0" else -10000
        for d in DEPTHS:
            arr = g.setdefault(f"evals_d{d}", [])
            pvs = g.setdefault(f"pv_d{d}", [])
            while len(arr) < n:
                arr.append(0)
            while len(pvs) < n:
                pvs.append("")
            if arr[-1] != final:
                arr[-1] = final
                pvs[-1] = f"{final}|#"
                fixed += 1
tmp = OUT + ".tmp"
with open(tmp, "w") as f:
    json.dump(doc, f)
os.replace(tmp, OUT)
lock.close()
print(f"final-ply mate fixed: {fixed} entries")
