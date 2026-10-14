#!/usr/bin/env python3
"""Materialize the KEPT, DEDUPED AJ/NIC games into PGN shards for the d10
labeling workers. Kept = movetext present AND at least one Elo tag
(amateur-vs-amateur dropped). Dedup = unique movetext hash across all
archives. Shards: N files of whole games, ~equal bytes."""
import glob
import hashlib
import os
import sys

BASE = "/srv/workspace/chess-active/aj-corpus"
OUT = "/srv/workspace/chess-active/aj-kept"
NSHARD = 96

os.makedirs(OUT, exist_ok=True)
seen = set()
kept = dup = noelo = nomoves = 0
shards = [open(f"{OUT}/aj.{i:03d}.pgn", "w") for i in range(NSHARD)]
sizes = [0] * NSHARD

for path in sorted(glob.glob(f"{BASE}/*/*.pgn")):
    cur_lines = []
    mov = []
    in_moves = False

    def flush():
        global kept, dup, noelo, nomoves
        if not cur_lines:
            return
        text = " ".join(mov).strip()
        hdrs = "".join(cur_lines)
        if not text or text in ("*", "1-0", "0-1", "1/2-1/2"):
            nomoves += 1
            return
        if "WhiteElo" not in hdrs and "BlackElo" not in hdrs:
            noelo += 1
            return
        h = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        if h in seen:
            dup += 1
            return
        seen.add(h)
        i = sizes.index(min(sizes))
        shards[i].write(hdrs + "\n" + text + "\n\n")
        sizes[i] += len(hdrs) + len(text)
        kept += 1

    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("[Event ") and in_moves:
            flush()
            cur_lines = []
            mov = []
            in_moves = False
        if line.startswith("["):
            cur_lines.append(line)
        else:
            t = line.strip()
            if t:
                mov.append(t)
                in_moves = True
    flush()

for f in shards:
    f.close()
print(f"kept {kept:,}  dup {dup:,}  noelo {noelo:,}  nomoves {nomoves:,}")
print("shard sizes MB: min", min(sizes) // 1048576, "max", max(sizes) // 1048576)
