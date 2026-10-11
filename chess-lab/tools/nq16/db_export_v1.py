#!/usr/bin/env python3
"""Training-set export v1 FROM THE DB (operator: 'just the fen in the DB').
Sources: seg_positions (OTB, game provenance) + nongame/puzzle (lichess).
Labels: DB d12 cp, stm-pov, verbatim (audited 81/100 stm vs fresh SF17).
Router: routerB 16-expert. Split: md5(fen)%20==0 -> val. Dedupe by fen.

Phases: prep (build dedup work table) | pack <i> <n> (parallel partial bins)
        | merge (concatenate + manifest)
"""
import glob
import hashlib
import os
import sqlite3
import struct
import sys

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess
from pack_expert_bins import pack_sfen, pack_move
from routerB import routeB

SRC = "/srv/workspace/chess-active/store/games.db"
WORK = "/srv/workspace/chess-active/store/export_v1.db"
OUT = "/srv/workspace/chess-active/bins-db-v1"
PART = OUT + "/parts"


def prep():
    if os.path.exists(WORK):
        os.remove(WORK)
    w = sqlite3.connect(WORK)
    w.execute("create table pos(fen text primary key, cp integer) without rowid")
    w.execute("attach database ? as g", (SRC,))
    n0 = w.execute("insert or ignore into pos select fen, cp from g.seg_positions").rowcount
    n1 = w.execute(
        "insert or ignore into pos select fen, cast(label as integer) from g.nongame "
        "where source='puzzle' and label glob '[-0-9]*'").rowcount
    w.commit()
    total = w.execute("select count(*) from pos").fetchone()[0]
    print(f"prep: seg {n0:,} inserted, puzzle {n1:,} inserted, dedup total {total:,}")
    # rowid map for parallel ranges (plain table -> implicit rowid)
    w.execute("drop table if exists posrow")
    w.execute("create table posrow as select fen, cp from pos")
    w.commit()
    print("prep done")


def pack(i, n):
    os.makedirs(PART, exist_ok=True)
    w = sqlite3.connect(f"file:{WORK}?mode=ro", uri=True)
    lo, hi = w.execute("select min(rowid), max(rowid) from posrow").fetchone()
    span = (hi - lo + 1 + n - 1) // n
    a, b = lo + i * span, min(lo + (i + 1) * span - 1, hi)
    files, counts = {}, {}
    skipped = 0
    for fen, cp in w.execute("select fen, cp from posrow where rowid between ? and ?", (a, b)):
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
        fh = files.setdefault((e, tag), open(f"{PART}/{e}.{tag}.{i}.bin", "wb"))
        fh.write(struct.pack("<32shHHbB", pack_sfen(board),
                             max(-30000, min(30000, int(cp))),
                             pack_move(mv), board.fullmove_number, 0, 0))
        counts[(e, tag)] = counts.get((e, tag), 0) + 1
    for fh in files.values():
        fh.close()
    print(f"pack[{i}]: rows {sum(counts.values()):,} skipped {skipped:,}")


def merge():
    os.makedirs(OUT, exist_ok=True)
    experts = sorted({os.path.basename(p).split(".")[0] for p in glob.glob(f"{PART}/*.bin")})
    total = 0
    print(f"{'expert':<14} {'train':>12} {'val':>10}")
    for e in experts:
        ntr = nva = 0
        for tag in ("train", "val"):
            dst = open(f"{OUT}/{e}.{tag}.bin", "wb")
            for srcf in sorted(glob.glob(f"{PART}/{e}.{tag}.*.bin")):
                with open(srcf, "rb") as sf:
                    while True:
                        chunk = sf.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                n = os.path.getsize(srcf) // 40
                if tag == "train":
                    ntr += n
                else:
                    nva += n
                os.remove(srcf)
            dst.close()
        total += ntr + nva
        print(f"{e:<14} {ntr:>12,} {nva:>10,}")
    print(f"TOTAL {total:,}")


if __name__ == "__main__":
    {"prep": lambda: prep(),
     "pack": lambda: pack(int(sys.argv[2]), int(sys.argv[3])),
     "merge": lambda: merge()}[sys.argv[1]]()
