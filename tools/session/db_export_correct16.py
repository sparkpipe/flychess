#!/usr/bin/env python3
"""Corrected 16-expert export — three sources, clean labels:
  1. OTB segments   (original eval-worker labels, proven healthy)
  2. miniatures     (SF17-d12 labels)
  3. puzzles        (nongame fens joined to the FRESH SF17-d12 relabels —
     replacing the corrupt solution-value labels)
Same router, same md5%20 split, same packer as db_export_v3."""
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
OUT = "/srv/workspace/chess-active/bins-correct16"

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
    e = routeB(fen)
    h = hashlib.md5(fen.encode()).hexdigest()
    tag = "val" if int(h, 16) % 20 == 0 else "train"
    key = (e, tag)
    if key not in files:
        files[key] = open(f"{OUT}/{e}.{tag}.bin", "wb")
    files[key].write(struct.pack("<32shHHbB", pack_sfen(board),
                                 max(-30000, min(30000, int(cp))),
                                 pack_move(mv), board.fullmove_number, 0, 0))
    counts[(e, tag, source)] = counts.get((e, tag, source), 0) + 1
    return "ok"


def scan(con, sql, source, label_every):
    n = 0
    for row in con.execute(sql):
        stats[emit(row[0], row[1], source)] += 1
        n += 1
        if n % label_every == 0:
            print(f"  {source} {n:,}", flush=True)
    print(f"{source} scanned: {n:,}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    scan(con, "select fen, cp from seg_positions", "otb_seg", 2000000)
    scan(con,
         "select p.fen, l.cp from positions p join labels l "
         "on l.fen=p.fen and l.engine='sf17' and l.depth=12 order by p.seq",
         "miniature", 500000)
    # puzzle pass: python-side join (the SQL join hit a pathological plan).
    # Batched IN-lookups against the labels PK — minutes instead of days.
    n = 0
    fens = [r[0] for r in con.execute("select fen from nongame")]
    print(f"puzzle fens to join: {len(fens):,}", flush=True)
    for i in range(0, len(fens), 900):
        batch = fens[i:i + 900]
        q = ("select fen, cp from labels where engine='sf17' and depth=12"
             " and fen in (%s)" % ",".join("?" * len(batch)))
        for fen, cp in con.execute(q, batch):
            stats[emit(fen, cp, "puzzle_sane")] += 1
            n += 1
        if (i // 900) % 50 == 0:
            print(f"  puzzle_sane joined {n:,}", flush=True)
    print(f"puzzle_sane scanned: {n:,}", flush=True)
    for fh in files.values():
        fh.close()
    print(f"emitted: {stats}", flush=True)

    with open(f"{OUT}/manifest.tsv", "w") as mf:
        mf.write("expert\ttrain\tval\totb_train\totb_val\tmini_train"
                 "\tmini_val\tpuz_train\tpuz_val\n")
        for e in sorted({k[0] for k in counts}):
            row = {"train": 0, "val": 0}
            for src, short in (("otb_seg", "otb"), ("miniature", "mini"),
                               ("puzzle_sane", "puz")):
                for tag in ("train", "val"):
                    c = counts.get((e, tag, src), 0)
                    row[tag] += c
                    row[f"{short}_{tag}"] = c
            mf.write(f"{e}\t{row['train']}\t{row['val']}\t"
                     f"{row.get('otb_train', 0)}\t{row.get('otb_val', 0)}\t"
                     f"{row.get('mini_train', 0)}\t{row.get('mini_val', 0)}\t"
                     f"{row.get('puz_train', 0)}\t{row.get('puz_val', 0)}\n")
    print("MANIFEST DONE", flush=True)


if __name__ == "__main__":
    main()
