#!/usr/bin/env python3
"""Build training bins at every depth from d10 to d20.
Same positions and sources as general3 — only the label depth changes."""
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
BASE = "/srv/workspace/chess-active/bins-curriculum"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
clean = json.load(open("/srv/workspace/chess-active/clean_short_wins.json"))
clean_gids = set(g["gid"] for g in clean)

# preload all d10-d20 labels into memory per fen (avoids 11 DB passes)
print("loading labels d10-d20...", flush=True)
labels = {}  # fen -> {depth: cp}
for fen, depth, cp in con.execute(
        "select fen, depth, cp from labels"
        " where engine = 'sf17' and depth between 10 and 20"):
    labels.setdefault(fen, {})[depth] = cp
print(f"fens with labels: {len(labels):,}", flush=True)

# preload gambit positions
gambit_fens = []
for line in open("/mnt/cold-raid6/chess-audit/gambit_aug_positions.txt"):
    p = line.strip().split("|")
    if p:
        gambit_fens.append(p[0])
print(f"gambit fens: {len(gambit_fens):,}", flush=True)

# preload clean win game positions
clean_fens = {}
for g in clean:
    plies = con.execute(
        "select fen from plies where gid = ? order by ply",
        (g["gid"],)).fetchall()
    clean_fens[g["gid"]] = [r[0] for r in plies]
print(f"clean win games: {len(clean_fens):,}", flush=True)

for D in range(10, 21):
    outdir = f"{BASE}/d{D}"
    os.makedirs(outdir, exist_ok=True)
    train_f = open(f"{outdir}/general3c.train.bin", "wb")
    val_f = open(f"{outdir}/general3c.val.bin", "wb")
    seen = set()
    counts = {"train": 0, "val": 0}

    def emit(fen, cp):
        try:
            board = chess.Board(fen)
            if not board.is_valid():
                return False
            mv = next(iter(board.legal_moves))
        except Exception:
            return False
        if fen in seen:
            return False
        seen.add(fen)
        h = hashlib.md5(fen.encode()).hexdigest()
        tag = "val" if int(h, 16) % 20 == 0 else "train"
        rec = struct.pack("<32shHHbB", pack_sfen(board),
                          max(-30000, min(30000, int(cp))),
                          pack_move(mv), board.fullmove_number, 0, 0)
        (train_f if tag == "train" else val_f).write(rec)
        counts[tag] += 1
        return True

    # OTB + puzzles (deduped)
    n = 0
    for fen, kind in con.execute(
            "select p.fen, p.kind from positions p"
            " where p.kind in ('otb_seg', 'puzzle')"):
        labs = labels.get(fen)
        if labs and D in labs:
            emit(fen, labs[D])
            n += 1
            if n % 2000000 == 0:
                print(f"  d{D} seg+puzzle {n:,}", flush=True)

    # clean wins
    for gid, fens in clean_fens.items():
        for fen in fens:
            labs = labels.get(fen)
            if labs and D in labs:
                emit(fen, labs[D])

    # gambit (duplicated, no dedup)
    ng = 0
    for fen in gambit_fens:
        labs = labels.get(fen)
        if labs and D in labs:
            try:
                board = chess.Board(fen)
                if not board.is_valid():
                    continue
                mv = next(iter(board.legal_moves))
            except Exception:
                continue
            h = hashlib.md5(fen.encode()).hexdigest()
            tag = "val" if int(h, 16) % 20 == 0 else "train"
            rec = struct.pack("<32shHHbB", pack_sfen(board),
                              max(-30000, min(30000, int(labs[D]))),
                              pack_move(mv), board.fullmove_number, 0, 0)
            (train_f if tag == "train" else val_f).write(rec)
            counts[tag] += 1
            ng += 1

    train_f.close()
    val_f.close()
    tr = os.path.getsize(f"{outdir}/general3c.train.bin") // 40
    va = os.path.getsize(f"{outdir}/general3c.val.bin") // 40
    print(f"d{D}: train {tr:,} val {va:,} (gambit dup: {ng:,})", flush=True)
print("ALL BINS DONE")
