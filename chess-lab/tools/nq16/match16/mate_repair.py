#!/usr/bin/env python3
"""One-shot repair: plies whose evals hold the old 1000+N mate encoding are
ambiguous with genuine 1000-2000cp evals — re-sweep those plies with SF17 and
rewrite evals_d*/pv_d* with the 10000+N encoding."""
import fcntl
import json
import os
import sys

sys.path.insert(0, "/srv/workspace/chess-active/matches")
from match_eval_worker import Engine  # noqa: E402

OUT = "/srv/workspace/chess-active/matches/games.json"
SF17 = "/srv/workspace/flychess/src/sf17/src/stockfish"
DEPTHS = [12, 20, 25]


def main():
    eng = Engine(SF17)
    lock = open(OUT + ".lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    doc = json.load(open(OUT))
    fixed = 0
    for m in doc:
        for g in m["games"]:
            for d in DEPTHS:
                arr = g.get(f"evals_d{d}")
                if not arr:
                    continue
                for i in range(len(arr)):
                    if abs(arr[i]) < 1000 or abs(arr[i]) > 2000:
                        continue
                    res = eng.top5(g["fens"][i], d)
                    if not res:
                        continue
                    g.setdefault(f"pv_d{d}", [""] * len(arr))[i] = \
                        " ;; ".join(f"{v}|{p}" for v, p in res)
                    arr[i] = res[0][0]
                    fixed += 1
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, OUT)
    lock.close()
    print(f"re-swept and rewrote {fixed} ambiguous eval entries")


if __name__ == "__main__":
    main()
