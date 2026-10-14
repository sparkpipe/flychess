#!/usr/bin/env python3
"""Ply-learnability: sample 1000 piece_down positions (mixed up/down),
train at each ply band (1-12, 13-24, ...) for fixed time, compare val_corr.
Actually per operator: train 1-ply data, 2-ply data, ... 12-ply data separately."""
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
OUT = "/srv/workspace/chess-active/plytest"
NNUE = "/srv/workspace/flychess/src/nnue-pytorch"
random.seed(17)

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
# piece_down positions with d12 labels, ply from FEN fullmove + stm
rows = con.execute(
    "select l.fen, l.cp from labels l where l.engine='sf17' and l.depth=12"
).fetchall()
random.shuffle(rows)
pd_rows = []
for fen, cp in rows:
    parts = fen.split()
    fm = int(parts[5])
    stm = 0 if parts[1] == "w" else 1
    ply = (fm - 1) * 2 + stm + 1
    try:
        b = chess.Board(fen)
        if not b.is_valid():
            continue
        if routeB(fen) != "piece_down":
            continue
    except Exception:
        continue
    pd_rows.append((fen, cp, ply))
    if len(pd_rows) >= 16000:
        break
print(f"sampled {len(pd_rows)} piece_down positions")

os.makedirs(OUT, exist_ok=True)
for target_ply in range(1, 13):
    tr = open(f"{OUT}/p{target_ply}.train.bin", "wb")
    va = open(f"{OUT}/p{target_ply}.val.bin", "wb")
    n_tr = n_va = 0
    for fen, cp, ply in pd_rows:
        if ply != target_ply:
            continue
        b = chess.Board(fen)
        mv = next(iter(b.legal_moves))
        import hashlib
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
    print(f"ply {target_ply}: train={n_tr} val={n_va}")
    if n_tr < 50:
        continue
    # train fixed 3 min
    run = f"{OUT}/run_p{target_ply}"
    os.makedirs(run, exist_ok=True)
    VR = min(2000, n_va)
    subprocess.run([
        sys.executable, "train.py", f"{OUT}/p{target_ply}.train.bin",
        "--validation-datasets", f"{OUT}/p{target_ply}.val.bin",
        "--validation-size", str(VR), "--check-val-every-n-epoch", "1",
        "--epoch-size", str(n_tr), "--batch-size", "512",
        "--max-time", "00:00:03:00", "--max-epochs", "100000",
        "--network-save-period", "100000", "--random-fen-skipping", "3",
        "--default-root-dir", run],
        cwd=NNUE, stdout=open(f"{run}/train.log", "w"),
        stderr=subprocess.STDOUT, timeout=400)
    # read best corr
    import csv, glob
    fs = sorted(glob.glob(f"{run}/lightning_logs/version_*/metrics.csv"))
    if fs:
        best = -1
        for r in csv.DictReader(open(fs[-1])):
            c = r.get("val_corr")
            if c not in (None, ""):
                try:
                    best = max(best, float(c))
                except ValueError:
                    pass
        print(f"ply {target_ply}: best val_corr = {best:.4f}")
