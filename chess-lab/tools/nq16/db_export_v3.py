#!/usr/bin/env python3
"""Export v3 — THE fleet training set. Unifies all labeled sources with
provenance: seg_positions (OTB) > nongame/puzzle > miniature positions.
Writes per-expert bins (routerB 16-expert, md5 val split), a manifest with
per-expert x per-source contributions, and sha256 of every bin.
Run ONLY after labeling is complete (asserted)."""
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

DB = "/srv/workspace/chess-active/store/games.db"
OUT = "/srv/workspace/chess-active/bins-fleet-v1"


def emit(fen, cp, source, ref, files, counts, seen):
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


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    # ASSERT: labeling complete
    pos = con.execute("select count(*) from positions").fetchone()[0]
    lab = con.execute(
        "select count(*) from labels where engine='sf17' and depth=12").fetchone()[0]
    unl = con.execute(
        "select count(*) from positions p where not exists("
        "select 1 from labels l where l.fen=p.fen and l.engine='sf17' and l.depth=12)"
    ).fetchone()[0]
    print(f"positions {pos:,}  sf17-d12 labels {lab:,}  unlabeled {unl:,}")
    if unl > 0:
        sys.exit("REFUSING: labeling incomplete — export only runs on complete data")
    os.makedirs(OUT, exist_ok=True)
    files, counts, seen = {}, {}, set()
    stats = {"ok": 0, "dup": 0, "invalid": 0}

    n = 0
    for fen, cp, gid, ply in con.execute(
            "select fen, cp, gid, ply from seg_positions"):
        stats[emit(fen, cp, "otb_seg", f"{gid}:{ply}", files, counts, seen)] += 1
        n += 1
        if n % 2000000 == 0:
            print(f"  seg {n:,}", flush=True)
    print(f"otb segments scanned: {n:,}")
    n = 0
    for fen, label in con.execute(
            "select fen, label from nongame where label glob '[-0-9]*'"):
        stats[emit(fen, int(label), "puzzle", "", files, counts, seen)] += 1
        n += 1
    print(f"puzzles scanned: {n:,}")
    n = 0
    for fen, cp in con.execute(
            "select p.fen, l.cp from positions p join labels l "
            "on l.fen=p.fen and l.engine='sf17' and l.depth=12 order by p.seq"):
        stats[emit(fen, cp, "miniature", "", files, counts, seen)] += 1
        n += 1
        if n % 5000000 == 0:
            print(f"  mini {n:,}", flush=True)
    print(f"miniatures scanned: {n:,}")
    for fh in files.values():
        fh.close()
    print(f"emitted: {stats}")

    # manifest: per expert x source x split + sha256
    experts = sorted({k[0] for k in counts})
    with open(f"{OUT}/manifest.tsv", "w") as mf:
        mf.write("expert\ttrain\tval\totb_seg_train\totb_seg_val\tpuzzle_train\t"
                 "puzzle_val\tminiature_train\tminiature_val\ttrain_sha256\tval_sha256\n")
        for e in experts:
            row = {"train": 0, "val": 0}
            for src in ("otb_seg", "puzzle", "miniature"):
                for tag in ("train", "val"):
                    c = counts.get((e, tag, src), 0)
                    row[tag] += c
                    row[f"{src}_{tag}"] = c
            th = hashlib.sha256(open(f"{OUT}/{e}.train.bin", "rb").read()).hexdigest()
            vh = hashlib.sha256(open(f"{OUT}/{e}.val.bin", "rb").read()).hexdigest()
            mf.write(f"{e}\t{row['train']}\t{row['val']}\t"
                     f"{row.get('otb_seg_train',0)}\t{row.get('otb_seg_val',0)}\t"
                     f"{row.get('puzzle_train',0)}\t{row.get('puzzle_val',0)}\t"
                     f"{row.get('miniature_train',0)}\t{row.get('miniature_val',0)}\t"
                     f"{th}\t{vh}\n")
    print("MANIFEST:", f"{OUT}/manifest.tsv")


if __name__ == "__main__":
    main()
