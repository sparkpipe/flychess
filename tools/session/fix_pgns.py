#!/usr/bin/env python3
"""Fix match PGNs: keep only the last N real games (the valid rerun),
dropping void 1-ply forfeit games from earlier crashed attempts."""
import re
import sys

def fix_pgn(path, keep=6):
    text = open(path, encoding="utf-8", errors="replace").read()
    # split into games on [Event
    blocks = re.split(r"(?=\[Event )", text)
    games = [b for b in blocks if b.strip().startswith("[Event")]
    if len(games) <= keep:
        print(f"{path}: {len(games)} games, <= {keep}, untouched")
        return
    kept = games[-keep:]
    # verify: kept games should have real move text (> 1 ply)
    for g in kept:
        moves = re.findall(r"\d+\.\s*\S+", g)
        if len(moves) < 3:
            print(f"WARNING: {path} kept-game has only {len(moves)} moves")
    out = "\n".join(kept)
    open(path, "w").write(out + "\n")
    print(f"{path}: {len(games)} -> {keep} games")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        fix_pgn(p)
