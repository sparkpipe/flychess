#!/usr/bin/env python3
"""Build per-depth train/val bins for the piece_down eval-depth experiment.
Positions and splits come from piece_down.fens (same set for every depth);
the label for depth D is read from the DB (labels, engine sf17, depth D).
Positions lacking a depth-D label are skipped and counted."""
import os, sqlite3, struct, sys
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess
from pack_expert_bins import pack_sfen, pack_move

DB = "/srv/workspace/chess-active/store/games.db"
BASE = "/srv/workspace/chess-active/pd-depthexp"
OUT = f"{BASE}/bins"
os.makedirs(OUT, exist_ok=True)

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
fens = [l.rstrip("\n").split("\t") for l in open(f"{BASE}/piece_down.fens") if l.strip()]
print(f"fen list: {len(fens):,}")

for D in range(1, 13):
    labs = dict(con.execute(
        "select fen, cp from labels where engine='sf17' and depth=?", (D,)))
    files = {t: open(f"{OUT}/d{D}.{t}.bin", "wb") for t in ("train", "val")}
    kept = {"train": 0, "val": 0}
    for fen, tag in fens:
        cp = labs.get(fen)
        if cp is None: continue
        try:
            board = chess.Board(fen)
            mv = next(iter(board.legal_moves))
        except Exception:
            continue
        files[tag].write(struct.pack("<32shHHbB", pack_sfen(board),
                         max(-30000, min(30000, int(cp))),
                         pack_move(mv), board.fullmove_number, 0, 0))
        kept[tag] += 1
    for fh in files.values(): fh.close()
    print(f"d{D}: train={kept[train]:,} val={kept[val]:,} (missing {len(fens)-kept[train]-kept[val]:,})")
