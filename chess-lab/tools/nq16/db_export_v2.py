#!/usr/bin/env python3
"""Export v2: training set from the DB positions+labels tables (ALL sources:
segments corpus, puzzles, miniatures — everything with an sf17-d12 label).
Router: routerB 16-expert. Split: md5(fen)%20==0 -> val. Deduped by fen PK."""
import hashlib
import os
import sqlite3
import struct
import sys

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess
from pack_expert_bins import pack_sfen, pack_move
from routerB import routeB

DB = "/srv/workspace/chess-active/store/games.db"
OUT = "/srv/workspace/chess-active/bins-db-v2"


def main():
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    n = con.execute(
        "select count(*) from positions p join labels l "
        "on l.fen=p.fen and l.engine='sf17' and l.depth=12").fetchone()[0]
    print(f"labeled positions: {n:,}")
    files, counts = {}, {}
    skipped = 0
    cur = con.execute(
        "select p.fen, l.cp from positions p join labels l "
        "on l.fen=p.fen and l.engine='sf17' and l.depth=12 order by p.seq")
    batch = []
    for fen, cp in cur:
        batch.append((fen, cp))
        if len(batch) >= 50000:
            emit(batch, files, counts)
            batch = []
    if batch:
        emit(batch, files, counts)
    for fh in files.values():
        fh.close()
    tot = 0
    print(f"{'expert':<14} {'train':>12} {'val':>10}")
    for (e, tag) in sorted(counts):
        pass
    tr = {}
    va = {}
    for (e, tag), c in counts.items():
        (tr if tag == "train" else va)[e] = c
    for e in sorted(tr):
        print(f"{e:<14} {tr[e]:>12,} {va.get(e, 0):>10,}")
        tot += tr[e] + va.get(e, 0)
    print(f"TOTAL {tot:,}  (skipped invalid {skipped})")


def emit(rows, files, counts):
    global skipped
    for fen, cp in rows:
        try:
            board = chess.Board(fen)
            if not board.is_valid():
                skipped += 1
                continue
            mv = next(iter(board.legal_moves))
        except Exception:
            skipped += 1
            continue
        e = routeB(fen)
        h = hashlib.md5(fen.encode()).hexdigest()
        tag = "val" if int(h, 16) % 20 == 0 else "train"
        key = (e, tag)
        if key not in files:
            files[key] = open(f"{OUT}/{e}.{tag}.bin", "wb")
        files[key].write(struct.pack("<32shHHbB", pack_sfen(board),
                                     max(-30000, min(30000, int(cp))),
                                     pack_move(mv), board.fullmove_number, 0, 0))
        counts[key] = counts.get(key, 0) + 1


if __name__ == "__main__":
    main()
