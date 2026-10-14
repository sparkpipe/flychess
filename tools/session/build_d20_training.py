#!/usr/bin/env python3
"""Build the d20 training set — general3:
  1. OTB segments with d20 labels
  2. Puzzles with d20 labels
  3. Miniatures with d20 labels
  4. Gambit winner positions (duplicated, d20 labels)
  5. Clean short wins (3,916 games, entire games, d20 labels)
"""
import hashlib
import json
import os
import sqlite3
import struct
import sys

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess
from pack_expert_bins import pack_sfen, pack_move

DB = "/srv/workspace/chess-active/store/games.db"
OUT = "/srv/workspace/chess-active/bins-general3"
os.makedirs(OUT, exist_ok=True)

files, counts, seen = {}, {}, set()
stats = {"ok": 0, "dup": 0, "invalid": 0}


def emit(fen, cp, source):
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            return "invalid"
        mv = next(iter(board.legal_moves))
    except Exception:
        return "invalid"
    if fen in seen:
        return "dup"
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

# 1-3: all sources with d20 labels (same query as correct16 but depth=20)
n = 0
for fen, cp in con.execute(
        "select p.fen, l.cp from positions p"
        " join labels l on l.fen = p.fen and l.engine = 'sf17' and l.depth = 20"
        " where p.kind in ('mini_game', 'otb_seg', 'puzzle', 'gambit')"):
    stats[emit(fen, cp, "d20_all")] += 1
    n += 1
    if n % 2000000 == 0:
        print(f"  d20_all {n:,}", flush=True)
print(f"d20-labeled positions: {n:,}", flush=True)

# 4: gambit augmentation — DUPLICATED (not deduped, by design)
n = 0
for line in open("/mnt/cold-raid6/chess-audit/gambit_aug_positions.txt"):
    p = line.strip().split("|")
    if not p:
        continue
    fen = p[0]
    r = con.execute(
        "select cp from labels where fen = ? and engine = 'sf17' and depth = 20",
        (fen,)).fetchone()
    if not r:
        continue
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            continue
        mv = next(iter(board.legal_moves))
    except Exception:
        continue
    h = hashlib.md5(fen.encode()).hexdigest()
    tag = "val" if int(h, 16) % 20 == 0 else "train"
    # DUPLICATE: no dedup check — gambit upweighting is deliberate
    files[tag].write(struct.pack("<32shHHbB", pack_sfen(board),
                                 max(-30000, min(30000, int(r[0]))),
                                 pack_move(mv), board.fullmove_number, 0, 0))
    counts[(tag, "gambit_aug")] = counts.get((tag, "gambit_aug"), 0) + 1
    n += 1
print(f"gambit augmented (duplicated): {n:,}", flush=True)

# 5: clean short wins — entire games
clean = json.load(open("/srv/workspace/chess-active/clean_short_wins.json"))
n = 0
for g in clean:
    plies = con.execute(
        "select fen, cp from plies p"
        " join labels l on l.fen = p.fen and l.engine = 'sf17' and l.depth = 20"
        " where p.gid = ? order by p.ply", (g["gid"],)).fetchall()
    for fen, cp in plies:
        try:
            board = chess.Board(fen)
            if not board.is_valid():
                continue
            mv = next(iter(board.legal_moves))
        except Exception:
            continue
        h = hashlib.md5(fen.encode()).hexdigest()
        tag = "val" if int(h, 16) % 20 == 0 else "train"
        # no dedup — these are selected games, deliberate upweighting
        files[tag].write(struct.pack("<32shHHbB", pack_sfen(board),
                                     max(-30000, min(30000, int(cp))),
                                     pack_move(mv), board.fullmove_number, 0, 0))
        counts[(tag, "clean_wins")] = counts.get((tag, "clean_wins"), 0) + 1
        n += 1
print(f"clean short win positions: {n:,}", flush=True)

for fh in files.values():
    fh.close()
print(f"emitted: {stats}", flush=True)

# manifest
with open(f"{OUT}/manifest.json", "w") as mf:
    tr = sum(v for (t, s), v in counts.items() if t == "train")
    va = sum(v for (t, s), v in counts.items() if t == "val")
    json.dump({
        "train_records": tr, "val_records": va,
        "sources": {f"{t}:{s}": v for (t, s), v in counts.items()},
        "train_sha256": hashlib.sha256(
            open(f"{OUT}/general3.train.bin", "rb").read()).hexdigest(),
        "val_sha256": hashlib.sha256(
            open(f"{OUT}/general3.val.bin", "rb").read()).hexdigest(),
    }, mf, indent=1)
    print(f"train: {tr:,}  val: {va:,}")
print("BUILD DONE")
