#!/usr/bin/env python3
"""Build d12 and d20 training bins for the smooth-rise (aj_sr) corpus.

Input: labels export TSV (fen<TAB>cp12<TAB>cp20) produced read-only from
games.db on the hub (pass-through). Records are the standard 40-byte bin
format, verbatim with build_curriculum_bins: struct "<32shHHbB" =
pack_sfen | clamped cp | pack_move(first legal) | fullmove | 0 | 0.
Split: md5(fen) % 20 == 0 -> val (5%), stable across builds.

Usage: build_smoothrise_bins.py --tsv FILE --out DIR
"""
import argparse
import hashlib
import os
import struct
import sys

for _p in ("chess-lab/tools", "tools"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", _p))
import chess
from pack_expert_bins import pack_sfen, pack_move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/d12", exist_ok=True)
    os.makedirs(f"{a.out}/d20", exist_ok=True)
    files = {
        (12, "train"): open(f"{a.out}/d12/train.bin", "wb"),
        (12, "val"): open(f"{a.out}/d12/val.bin", "wb"),
        (20, "train"): open(f"{a.out}/d20/train.bin", "wb"),
        (20, "val"): open(f"{a.out}/d20/val.bin", "wb"),
    }
    counts = {(12, "train"): 0, (12, "val"): 0, (20, "train"): 0, (20, "val"): 0}
    seen = set()
    bad = dup = 0
    for line in open(a.tsv):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        fen, c12, c20 = parts
        try:
            board = chess.Board(fen)
            if not board.is_valid():
                bad += 1
                continue
            mv = next(iter(board.legal_moves))
        except Exception:
            bad += 1
            continue
        if fen in seen:
            dup += 1
            continue
        seen.add(fen)
        tag = "val" if int(hashlib.md5(fen.encode()).hexdigest(), 16) % 20 == 0 else "train"
        for depth, cp in ((12, int(c12)), (20, int(c20))):
            rec = struct.pack("<32shHHbB", pack_sfen(board),
                              max(-30000, min(30000, cp)),
                              pack_move(mv), board.fullmove_number, 0, 0)
            files[(depth, tag)].write(rec)
            counts[(depth, tag)] += 1
    for f in files.values():
        f.close()
    print(f"unique fens: {len(seen):,} (dup {dup:,}, bad {bad:,})")
    for k in sorted(counts):
        print(f"  d{k[0]} {k[1]}: {counts[k]:,}")


if __name__ == "__main__":
    main()
