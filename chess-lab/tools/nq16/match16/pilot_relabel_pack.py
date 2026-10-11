#!/usr/bin/env python3
"""PILOT: piece_down expert with fully-correct data (SF17 d12, stm-pov).
Sample piece_down-routed corpus positions -> relabel SF17 -> pack bin -> ready to train.
Proves the corrected pipeline end-to-end before committing to full retraining."""
import glob
import hashlib
import multiprocessing as mp
import os
import random
import struct
import subprocess
import sys

sys.path.insert(0, "/tmp")
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess
from pack_expert_bins import pack_sfen, pack_move
from routerB import routeB

SF17 = "/srv/workspace/flychess/src/sf17/src/stockfish"
OUT = "/srv/workspace/chess-active/pilot_pd"
TARGET = 60000
random.seed(23)


def collect():
    """piece_down-routed FENs from puzzle shards + segments (real corpus only)."""
    fens = []
    for sh in sorted(glob.glob("/home/spec/backlog_shards/puzzle_eval/w*.tsv")):
        for line in open(sh, errors="replace"):
            f = line.split("\t")[0]
            fens.append(f)
            if len(fens) > 400000:
                break
        if len(fens) > 400000:
            break
    segs = sorted(glob.glob("/mnt/cold-raid6/chess-audit/wp_fit/segments/pos_*.tsv"))
    for s in segs[:6]:
        for line in open(s, errors="replace"):
            fens.append(line.split("|")[0])
            if len(fens) > 800000:
                break
        if len(fens) > 800000:
            break
    random.shuffle(fens)
    out = []
    for f in fens:
        try:
            b = chess.Board(f)
            if not b.is_valid():
                continue
            if routeB(f) == "piece_down":
                out.append(f)
        except Exception:
            continue
        if len(out) >= TARGET:
            break
    return out


def relabel_worker(fens):
    """SF17 d12, score kept stm-pov (the bin convention). Returns [(fen, cp)]."""
    p = subprocess.Popen([SF17], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\nisready\n")
    p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    res = []
    for fen in fens:
        p.stdin.write(f"position fen {fen}\ngo depth 12\n")
        p.stdin.flush()
        last = None
        ok = True
        while True:
            l = p.stdout.readline()
            if not l:
                ok = False
                break
            if l.startswith("info ") and " score cp " in l and " pv " in l:
                t = l.split()
                for i, x in enumerate(t):
                    if x == "cp":
                        last = int(t[i + 1])
            elif l.startswith("bestmove"):
                break
        if ok and last is not None:
            res.append((fen, last))
    p.stdin.write("quit\n")
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    src = f"{OUT}/sources.txt"
    if os.path.exists(src):
        fens = [l.strip() for l in open(src) if l.strip()]
        print(f"reusing {len(fens):,} collected")
    else:
        fens = collect()
        open(src, "w").write("\n".join(fens))
        print(f"collected {len(fens):,} piece_down positions")
    NW = 20
    chunks = [fens[i::NW] for i in range(NW)]
    with mp.Pool(NW) as pool:
        parts = pool.map(relabel_worker, chunks)
    labeled = [x for part in parts for x in part]
    print(f"labeled {len(labeled):,}/{len(fens):,} with SF17 d12 (stm-pov)")
    with open(f"{OUT}/labels17.tsv", "w") as f:
        for fen, cp in labeled:
            f.write(f"{fen}\t{cp}\n")
    # pack: stm-pov labels, fixed convention, 95/5 split by fen hash
    tr = open(f"{OUT}/pd.train.bin", "wb")
    va = open(f"{OUT}/pd.val.bin", "wb")
    n_tr = n_va = 0
    for fen, cp in labeled:
        try:
            board = chess.Board(fen)
            mv = next(iter(board.legal_moves))
        except Exception:
            continue
        rec = struct.pack("<32shHHbB", pack_sfen(board),
                          max(-30000, min(30000, int(cp))),
                          pack_move(mv), board.fullmove_number, 0, 0)
        h = int(hashlib.md5(fen.encode()).hexdigest(), 16)
        if h % 20 == 0:
            va.write(rec); n_va += 1
        else:
            tr.write(rec); n_tr += 1
    tr.close(); va.close()
    print(f"PILOT BIN READY: train {n_tr:,} val {n_va:,} at {OUT}")


if __name__ == "__main__":
    main()
