#!/usr/bin/env python3
"""Build general20 — the approved manifest, verbatim sources:

  1. nQ structural bins (expert_bins_both/*.bin, all 22 incl. aug/gambit)
  2. tactics train/val bins (nQ's own split)
  3. d20 eval-slope segments (extractor-v2 rules on the d20 corpus,
     md5%20 val split, stm-pov d20 labels)
  4. DEGM degm.tsv (cp/100, white-pov -> stm at pack)
  5. syzygy tb.tsv (wdl -> material-aware wp-inversion cp, stm verbatim)

No dedup, no filtering, no relabeling. Writes manifest + sha256."""
import glob
import hashlib
import json
import math
import os
import sqlite3
import struct
import sys

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess
from pack_expert_bins import pack_sfen, pack_move

EB = "/mnt/cold-raid6/chess-audit/expert_bins_both"
TACT = "/mnt/cold-raid6/chess-audit/QUARANTINE_DUMP_tactics_{t}.bin"
DEGM = "/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/degm.tsv"
TBTSV = "/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/tb.tsv"
DB = "/srv/workspace/chess-active/store/games.db"
OUT = "/srv/workspace/chess-active/bins-general20"
os.makedirs(OUT, exist_ok=True)

counts = {}


def rec(board, cp):
    mv = next(iter(board.legal_moves))
    return struct.pack("<32shHHbB", pack_sfen(board),
                       max(-30000, min(30000, int(cp))), pack_move(mv),
                       board.fullmove_number, 0, 0)


def shard(fen):
    return "val" if int(hashlib.md5(fen.encode()).hexdigest(), 16) % 20 == 0 \
        else "train"


# --- 1+2: byte concatenation of nQ bins ---
tr = open(f"{OUT}/general20.train.bin", "wb")
va = open(f"{OUT}/general20.val.bin", "wb")
for f in sorted(glob.glob(f"{EB}/*.bin")):
    n = os.path.getsize(f) // 40
    tr.write(open(f, "rb").read())
    name = os.path.basename(f)[:-4]
    counts[f"nq:{name}"] = n
for t, fh in (("train", tr), ("val", va)):
    f = TACT.format(t=t)
    counts[f"nq:tactics_{t}"] = os.path.getsize(f) // 40
    fh.write(open(f, "rb").read())

# --- 3: d20 eval-slope segments ---
wp_of = lambda cp: 1.0 / (1.0 + math.exp(-cp / 361.0))


def band(wp):
    if wp < 0.30:
        return "collapse"
    if wp < 0.45:
        return "defend"
    if wp < 0.55:
        return "equalize"
    if wp < 0.70:
        return "press"
    if wp < 0.95:
        return "convert"
    return "win"


con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
games = {g: (o, r) for g, o, r in
         con.execute("select gid, off, result from games")}
rows = con.execute(
    "select p.ref, p.fen, l.cp from positions p"
    " join labels l on l.fen = p.fen and l.engine='sf17' and l.depth=20"
    " order by p.seq").fetchall()
by_game = {}
for ref, fen, cp in rows:
    gid, ply = ref.split(":")
    by_game.setdefault(int(gid), []).append((int(ply), fen, cp))
n_seg_pos = {"train": 0, "val": 0}
for gid, plies in by_game.items():
    off, result = games.get(gid, (0, "1/2-1/2"))
    if result not in ("1-0", "0-1"):
        continue
    winner_is_white = result == "1-0"
    plies.sort()
    seq = []
    for ply, fen, cp in plies:
        stm_white = fen.split()[1] == "w"
        seq.append((fen, cp, cp if (stm_white == winner_is_white) else -cp))
    segs, cur = [], [seq[0]]
    for prev, nxt in zip(seq, seq[1:]):
        if abs(nxt[2] - prev[2]) >= 120:
            segs.append(cur)
            cur = []
        cur.append(nxt)
    segs.append(cur)
    for seg in segs:
        if band(wp_of(seg[0][2])) == band(wp_of(seg[-1][2])):
            continue                    # same-band: draw-to-draw rule
        for fen, cp_stm, _w in seg:
            try:
                board = chess.Board(fen)
                if not board.is_valid():
                    continue
                r = rec(board, cp_stm)  # stm-pov verbatim (bin law)
            except Exception:
                continue
            s = shard(fen)
            (tr if s == "train" else va).write(r)
            n_seg_pos[s] += 1
counts["d20_segments:train"] = n_seg_pos["train"]
counts["d20_segments:val"] = n_seg_pos["val"]

# --- 4: DEGM (white-pov cp/100 -> stm) ---
n = 0
for line in open(DEGM, encoding="utf-8", errors="replace"):
    p = line.rstrip("\n").split("|")
    if len(p) < 4:
        continue
    try:
        board = chess.Board(p[0])
        cp = int(p[2]) / 100.0
        cp_stm = cp if board.turn == chess.WHITE else -cp
        r = rec(board, cp_stm)
    except Exception:
        continue
    (tr if shard(p[0]) == "train" else va).write(r)
    n += 1
counts["degm"] = n

# --- 5: syzygy tb.tsv (wdl -> material-aware wp-inversion, stm verbatim) ---


def win_prob(cp, mat):
    m = min(max(mat, 17), 78) / 58.0
    a = -142.72052667 + m * (372.35176398 + m * (-340.71073572 + m * 415.23490212))
    b = 5.93832785 + m * (15.61267078 + m * (-30.57816876 + m * 69.63866711))
    return 1.0 / (1.0 + pow(10.0, -cp / (400.0 * b / (a + b) / 0.1817)))


def cp_for_win(mat, target=0.90):
    lo, hi = 0.0, 2000.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if win_prob(mid, mat) < target:
            lo = mid
        else:
            hi = mid
    return int((lo + hi) / 2)


n = 0
for line in open(TBTSV):
    p = line.rstrip("\n").split("|")
    fen, wdl = p[0], int(p[2])
    try:
        board = chess.Board(fen)
        mat = sum({"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}[pc.symbol().upper()]
                  for pc in board.piece_map().values()
                  if pc.symbol().upper() != "K")
        cp = cp_for_win(mat) if wdl > 0 else (-cp_for_win(mat) if wdl < 0 else 0)
        r = rec(board, cp)
    except Exception:
        continue
    (tr if shard(fen) == "train" else va).write(r)
    n += 1
counts["syzygy"] = n

tr.close()
va.close()

# manifest + sha256
man = {"counts": counts,
       "train_records": os.path.getsize(f"{OUT}/general20.train.bin") // 40,
       "val_records": os.path.getsize(f"{OUT}/general20.val.bin") // 40}
for t in ("train", "val"):
    man[f"{t}_sha256"] = hashlib.sha256(
        open(f"{OUT}/general20.{t}.bin", "rb").read()).hexdigest()
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
print(json.dumps(man, indent=1))
