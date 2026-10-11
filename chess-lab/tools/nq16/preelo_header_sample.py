#!/usr/bin/env python3
"""Fast pre-Elo sample via raw header scan (no move parsing).
Shows qualifying-candidate games: no Elo, decisive, winner recognized.
Length rule verified in the real builder; this is for population approval."""
import re
from collections import Counter, defaultdict

PGN = "/mnt/cold-raid6/rtx5090-archive/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn"
CENSUS = "/srv/workspace/chess-active/players_census.tsv"
MIN_GAMES = 100

rec = {}
for line in open(CENSUS, encoding="utf-8", errors="replace"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        rec[p[0]] = (int(p[1]), p[2] if len(p) > 2 else "")


def recognized(name):
    r = rec.get(name)
    return r is not None and (r[0] >= MIN_GAMES or bool(r[1]))


tag = re.compile(r'^\[(\w+)\s+"(.*)"\]$')
hdr = {}
shown = 0
total_preelo = Counter()
with open(PGN, encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("["):
            m = tag.match(line.strip())
            if m:
                hdr[m.group(1)] = m.group(2)
            continue
        if not (line.startswith("1.") or line.startswith("1 ")):
            continue
        h = hdr
        hdr = {}
        if "WhiteElo" in h or "BlackElo" in h:
            continue
        res = h.get("Result", "")
        if res not in ("1-0", "0-1"):
            continue
        wname = h.get("White" if res == "1-0" else "Black", "?")
        total_preelo["decisive_preelo"] += 1
        if not recognized(wname):
            continue
        total_preelo["recognized_winner"] += 1
        if shown < 12:
            lname = h.get("Black" if res == "1-0" else "White", "?")
            r = rec.get(wname, (0, ""))
            year = h.get("Date", h.get("EventDate", "?"))[:4]
            print(f"{wname} ({r[0]} games{',' + r[1] if r[1] else ''}) beat "
                  f"{lname} — {res}, {year} {h.get('Event', '?')[:30]}")
            shown += 1
print(dict(total_preelo))
