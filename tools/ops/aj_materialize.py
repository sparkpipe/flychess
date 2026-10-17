#!/usr/bin/env python3
"""Materialize the KEPT, DEDUPED AJ/NIC games into PGN shards for the d10
labeling workers. Kept = movetext present AND at least one Elo tag.
Dedup = unique movetext hash across all archives. Shards: N files of whole
games, ~equal bytes. Deterministic. Verbatim port of the original builder
with parameterized paths + optional --only regeneration.

Usage: aj_materialize.py --corpus DIR --out DIR [--nshard 96]
                          [--only 012,013,...]
"""
import argparse
import glob
import hashlib
import os

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--nshard", type=int, default=96)
ap.add_argument("--only", default="",
                help="comma-separated shard numbers to (re)generate; "
                     "assignment unchanged, others not written")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
only = {int(x) for x in args.only.split(",") if x.strip()} if args.only else None
NSHARD = args.nshard
seen = set()
kept = dup = noelo = nomoves = 0
shards = {}
sizes = [0] * NSHARD


def file_for(i):
    if only is not None and i not in only:
        return None
    f = shards.get(i)
    if f is None:
        f = open(f"{args.out}/aj.{i:03d}.pgn", "w")
        shards[i] = f
    return f



seen = set()
kept = dup = noelo = nomoves = 0

sizes = [0] * NSHARD

for path in sorted(glob.glob(f"{args.corpus}/*/*.pgn")):
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
        f = file_for(i)
        if f is not None:
            f.write(hdrs + "\n" + text + "\n\n")
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

for f in shards.values():
    f.close()
print(f"kept {kept:,}  dup {dup:,}  noelo {noelo:,}  nomoves {nomoves:,}")
print("shard sizes MB: min", min(sizes) // 1048576, "max", max(sizes) // 1048576)
