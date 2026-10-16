#!/usr/bin/env python3
"""Build general3 CORRECTLY:
  - OTB segments with d20 labels (not the problem)
  - Puzzles with d20 labels (not the problem)
  - Miniatures: ONLY the 3,916 clean short wins (~75K positions)
  - Gambit winner positions duplicated with d20 labels
"""
import argparse
import hashlib
import json
import os
import sqlite3
import struct
import sys

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--db", required=True, help="games.db (read-only)")
ap.add_argument("--out", required=True, help="output dir for general3 bins")
ap.add_argument("--clean-wins", required=True, help="clean_short_wins.json")
ap.add_argument("--gambit", required=True, help="gambit_aug_positions.txt")
args = ap.parse_args()
DB, OUT = args.db, args.out

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "chess-lab", "tools"))
import chess
from pack_expert_bins import pack_sfen, pack_move

os.makedirs(OUT, exist_ok=True)

files, counts, seen = {}, {}, set()
stats = {"ok": 0, "dup": 0, "invalid": 0}


def emit(fen, cp, source, dedup=True):
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            return "invalid"
        mv = next(iter(board.legal_moves))
    except Exception:
        return "invalid"
    if dedup and fen in seen:
        return "dup"
    if dedup:
        seen.add(fen)
    h = hashlib.md5(fen.encode()).hexdigest()
    tag = "val" if int(h, 16) % 20 == 0 else "train"
    if tag not in files:
        files[tag] = open(f"{OUT}/general3.{tag}.bin", "wb")
    files[tag].write(struct.pack("<32shHHbB", pack_sfen(board),
                                 max(-30000, min(30000, int(cp))),
                                 pack_move(mv), board.fullmove_number, 0, 0))
    counts[(tag, source)] = counts.get((tag, source), 0) + 1
    return "ok"


con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

# clean short win gids (the ONLY miniature games we keep)
clean = json.load(open(args.clean_wins))
clean_gids = set(g["gid"] for g in clean)
print(f"clean short win gids: {len(clean_gids):,}")

# 1: OTB segments + puzzles with d20 labels
n = 0
for fen, cp, kind in con.execute(
        "select p.fen, l.cp, p.kind from positions p"
        " join labels l on p.fen = l.fen and l.engine = 'sf17' and l.depth = 20"
        " where p.kind in ('otb_seg', 'puzzle')"):
    stats[emit(fen, cp, kind)] += 1
    n += 1
    if n % 2000000 == 0:
        print(f"  seg+puzzle {n:,}", flush=True)
print(f"OTB + puzzles (d20): {n:,}", flush=True)

# 2: miniatures — ONLY the clean short wins
n = 0
for g in clean:
    plies = con.execute(
        "select p.fen, l.cp from plies p"
        " join labels l on p.fen = l.fen"
        "   and l.engine = 'sf17' and l.depth = 20"
        " where p.gid = ? order by p.ply", (g["gid"],)).fetchall()
    for fen, cp in plies:
        stats[emit(fen, cp, "clean_wins")] += 1
        n += 1
print(f"clean short wins (d20, full games): {n:,}", flush=True)

# 3: gambit duplication (deliberately NOT deduped)
n = 0
for line in open(args.gambit):
    p = line.strip().split("|")
    if not p:
        continue
    fen = p[0]
    r = con.execute(
        "select cp from labels where fen = ? and engine = 'sf17' and depth = 20",
        (fen,)).fetchone()
    if not r:
        continue
    emit(fen, r[0], "gambit_aug", dedup=False)
    n += 1
print(f"gambit augmented (duplicated): {n:,}", flush=True)

for fh in files.values():
    fh.close()

# manifest
tr = sum(v for (t, s), v in counts.items() if t == "train")
va = sum(v for (t, s), v in counts.items() if t == "val")
import hashlib as h
man = {
    "train_records": tr, "val_records": va,
    "sources": {f"{t}:{s}": v for (t, s), v in counts.items()},
    "train_sha256": h.sha256(open(f"{OUT}/general3.train.bin", "rb").read()).hexdigest(),
    "val_sha256": h.sha256(open(f"{OUT}/general3.val.bin", "rb").read()).hexdigest(),
}
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
print(f"TRAIN: {tr:,}  VAL: {va:,}  TOTAL: {tr+va:,}")
print(json.dumps(man["sources"], indent=1))
