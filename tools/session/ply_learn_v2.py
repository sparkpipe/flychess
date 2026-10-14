#!/usr/bin/env python3
"""Ply-learnability v2: seg_positions (ply 1+) + DB positions, piece_down only,
500+500 up/down mixed, fixed 3min per ply band 1-12 + 20s/40s baselines."""
import csv
import glob
import hashlib
import os
import random
import sqlite3
import struct
import subprocess
import sys

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess
from pack_expert_bins import pack_sfen, pack_move
from routerB import routeB

DB = "/srv/workspace/chess-active/store/games.db"
OUT = "/srv/workspace/chess-active/plytest2"
NNUE = "/srv/workspace/flychess/src/nnue-pytorch"
random.seed(17)

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
pd_by_ply = {}
# segment positions: ply 1-14 (need early plies) routed to piece_down
for fen, cp in con.execute(
        "select fen, cp from seg_positions where cast(substr(fen,"
        " length(fen) - instr(reverse(fen), reverse(' ')) + 2) as integer)"
        " <= 8 order by random() limit 200000"):
    parts = fen.split()
    try:
        fm = int(parts[5])
    except (ValueError, IndexError):
        continue
    stm = 0 if parts[1] == "w" else 1
    ply = (fm - 1) * 2 + stm + 1
    if ply > 14:
        continue
    try:
        b = chess.Board(fen)
        if not b.is_valid() or routeB(fen) != "piece_down":
            continue
    except Exception:
        continue
    pd_by_ply.setdefault(ply, []).append((fen, cp, b))

print("seg-source ply counts:", {k: len(v) for k, v in sorted(pd_by_ply.items())})

os.makedirs(OUT, exist_ok=True)


def run_band(label, rows):
    tr = open(f"{OUT}/{label}.train.bin", "wb")
    va = open(f"{OUT}/{label}.val.bin", "wb")
    n_tr = n_va = 0
    for fen, cp, b in rows:
        mv = next(iter(b.legal_moves))
        h = hashlib.md5(fen.encode()).hexdigest()
        f = va if int(h, 16) % 5 == 0 else tr
        f.write(struct.pack("<32shHHbB", pack_sfen(b),
                            max(-30000, min(30000, int(cp))),
                            pack_move(mv), b.fullmove_number, 0, 0))
        if f is tr:
            n_tr += 1
        else:
            n_va += 1
    tr.close()
    va.close()
    if n_tr < 50:
        print(f"{label}: SKIP (train={n_tr})")
        return
    run = f"{OUT}/run_{label}"
    os.makedirs(run, exist_ok=True)
    subprocess.run(
        [sys.executable, "train.py", f"{OUT}/{label}.train.bin",
         "--validation-datasets", f"{OUT}/{label}.val.bin",
         "--validation-size", str(min(2000, n_va)),
         "--check-val-every-n-epoch", "1", "--epoch-size", str(n_tr),
         "--batch-size", "512", "--max-time", "00:00:03:00",
         "--max-epochs", "100000", "--network-save-period", "100000",
         "--random-fen-skipping", "3", "--default-root-dir", run],
        cwd=NNUE, stdout=open(f"{run}/train.log", "w"),
        stderr=subprocess.STDOUT, timeout=400)
    best = -1
    fs = sorted(glob.glob(f"{run}/lightning_logs/version_*/metrics.csv"))
    if fs:
        for r in csv.DictReader(open(fs[-1])):
            c = r.get("val_corr")
            if c not in (None, ""):
                try:
                    best = max(best, float(c))
                except ValueError:
                    pass
    print(f"{label}: train={n_tr} val={n_va} best_corr={best:.4f}", flush=True)


for ply in list(range(1, 15)) + [20, 30, 40]:
    rows = pd_by_ply.get(ply, [])
    random.shuffle(rows)
    run_band(f"ply{ply}", rows[:1000])
