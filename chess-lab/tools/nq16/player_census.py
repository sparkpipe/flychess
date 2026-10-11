#!/usr/bin/env python3
"""Player census from PGN headers (fast raw scan, no move parsing):
- game count per player name (white+black)
- titles per player (WhiteTitle/BlackTitle headers)
Output: players_census.tsv  name <tab> games <tab> title"""
import re
from collections import Counter, defaultdict

PGN = "/mnt/cold-raid6/rtx5090-archive/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn"
OUT = "/srv/workspace/chess-active/players_census.tsv"

games = Counter()
titles = defaultdict(set)
hdr = {}
n = 0
tag = re.compile(r'^\[(\w+)\s+"(.*)"\]$')
with open(PGN, encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("["):
            m = tag.match(line.strip())
            if m:
                hdr[m.group(1)] = m.group(2)
            continue
        if line.startswith("1.") or (line.startswith("1 ") and hdr):
            w = hdr.get("White", "?")
            b = hdr.get("Black", "?")
            games[w] += 1
            games[b] += 1
            wt = hdr.get("WhiteTitle", "")
            bt = hdr.get("BlackTitle", "")
            if wt:
                titles[w].add(wt)
            if bt:
                titles[b].add(bt)
            n += 1
            hdr = {}
print(f"games scanned: {n:,}, players: {len(games):,}, titled: {len(titles):,}",
      flush=True)
with open(OUT, "w") as f:
    for name, cnt in games.most_common():
        f.write(f"{name}\t{cnt}\t{','.join(sorted(titles[name]))}\n")
print("census written:", OUT)
