#!/usr/bin/env python3
"""Bake d12 evals for EVERY ply of EVERY game into the BOX games.json
(the source of truth — serve.py rsyncs it down every 12s, so anything
baked locally is clobbered; bake upstream instead).
Idempotent: skips positions that already have a value. White-pov.
Usage: python3 d12_bake.py"""
import json
import subprocess
import urllib.request

BOX = "rtx5090"
REMOTE = "/srv/workspace/chess-active/matches/games.json"
LOCAL = "/tmp/games_bake.json"
SVC = "http://127.0.0.1:8124/eval"


def ev(fen, depth=12):
    req = urllib.request.Request(
        SVC, data=json.dumps({"fen": fen, "depth": depth}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["cp"]


def main():
    subprocess.run(["scp", "-q", f"{BOX}:{REMOTE}", LOCAL], check=True)
    data = json.load(open(LOCAL))
    total = missing = 0
    for m in data:
        for g in m["games"]:
            a = g.get("evals_d12")
            if not isinstance(a, list) or len(a) != len(g["fens"]):
                a = [None] * len(g["fens"])
                g["evals_d12"] = a
            for i, fen in enumerate(g["fens"]):
                total += 1
                if a[i] in (None, 0):
                    a[i] = ev(fen)
                    missing += 1
    with open(LOCAL, "w") as f:
        json.dump(data, f)
    subprocess.run(["scp", "-q", LOCAL, f"{BOX}:{REMOTE}"], check=True)
    print(f"positions {total:,}  freshly evaluated {missing:,}  pushed to box")


if __name__ == "__main__":
    main()
