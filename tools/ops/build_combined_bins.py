#!/usr/bin/env python3
"""Build the COMBINED training bins: old general3 sources + new smooth-rise
(aj_sr) corpus — operator ruling 2026-10-23: train old and new together.

Composition verbatim with build_general3_v2 plus aj_sr:
  1. otb_seg + puzzle positions (deduped)            [old]
  2. clean-short-win full games via plies (deduped)  [old]
  3. gambit-aug positions, DUPLICATED (no dedup)     [old, deliberate mass]
  4. aj_sr smooth-rise positions (deduped)           [new]
Emitted at BOTH depth 12 and depth 20 for the two-phase recipe.
Record format: struct "<32shHHbB" (40B), md5(fen)%20==0 -> val (5%).

Inputs: --pos P_positions.tsv --plies P_plies.tsv --gambit gambit_file --out DIR
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
    ap.add_argument("--pos", required=True)
    ap.add_argument("--plies", required=True)
    ap.add_argument("--gambit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for d in ("d12", "d20"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    files, counts = {}, {}
    seen = set()
    stats = {"ok": 0, "dup": 0, "invalid": 0}

    def emit(fen, c12, c20, source, dedup=True):
        try:
            board = chess.Board(fen)
            if not board.is_valid():
                return "invalid"
            mv = next(iter(board.legal_moves))
        except Exception:
            return "invalid"
        if dedup and fen in seen:
            return "dup"
        if dedup:
            seen.add(fen)
        h = hashlib.md5(fen.encode()).hexdigest()
        tag = "val" if int(h, 16) % 20 == 0 else "train"
        for depth, cp in ((12, c12), (20, c20)):
            k = (depth, tag)
            if k not in files:
                files[k] = open(f"{a.out}/d{depth}/{tag}.bin", "wb")
            files[k].write(struct.pack("<32shHHbB", pack_sfen(board),
                                       max(-30000, min(30000, int(cp))),
                                       pack_move(mv), board.fullmove_number, 0, 0))
        counts[(tag, source)] = counts.get((tag, source), 0) + 1
        return "ok"

    # gambit file fens need label lookup — preload from the positions export
    gambit_cp = {}
    n = 0
    for line in open(a.pos):
        kind, fen, c12, c20 = line.rstrip("\n").split("\t")
        if kind in ("gambit", "fill"):
            gambit_cp[fen] = (c12, c20)
        n += 1
        if n % 4000000 == 0:
            print(f"  pos-pass {n:,}", flush=True)
    print(f"positions export read: {n:,} (gambit rows {len(gambit_cp):,})", flush=True)

    # 1: otb_seg + puzzle + aj_sr (deduped)
    n = 0
    pos = open(a.pos)
    for line in pos:
        kind, fen, c12, c20 = line.rstrip("\n").split("\t")
        if kind in ("gambit", "fill"):
            continue          # fill flows via plies/gambit-file paths
        stats[emit(fen, c12, c20, kind)] += 1
        n += 1
        if n % 4000000 == 0:
            print(f"  seg/puzzle/aj_sr {n:,}", flush=True)
    print(f"otb_seg + puzzle + aj_sr: {n:,}", flush=True)

    # 2: clean short wins via plies
    n = 0
    for line in open(a.plies):
        gid, ply, fen, c12, c20 = line.rstrip("\n").split("\t")
        stats[emit(fen, c12, c20, "clean_wins")] += 1
        n += 1
    print(f"clean short wins: {n:,}", flush=True)

    # 3: gambit duplication (deliberately NOT deduped)
    n = miss = 0
    for gline in open(a.gambit):
        fen = gline.strip().split("|")[0]
        if not fen:
            continue
        cps = gambit_cp.get(fen)
        if not cps:
            miss += 1
            continue
        emit(fen, cps[0], cps[1], "gambit_aug", dedup=False)
        n += 1
    print(f"gambit duplicated: {n:,} (label-missing {miss:,})", flush=True)

    for f in files.values():
        f.close()
    print(f"unique fens: {len(seen):,}; stats: {stats}")
    for (tag, source), v in sorted(counts.items()):
        print(f"  {tag} {source}: {v:,}")


if __name__ == "__main__":
    main()
