"""Pack AUDITED per-expert training bins from segment_extractor_v2 output.

Rules (DESIGN-MOE v1.10):
- Selection: ANY band crossing = demonstrated improvement by someone (same-band out).
- Expert cascade: TB-region <=5 men (counted, NOT packed - TB expert = syzygy data);
  11 residue confrontation classes; dvoretsky 6-10 men; exchanges (dual membership:
  positions where the material category changed ALSO pack into their stable class);
  gambit (ply<24 asymmetric); balanced -> shards by lock_c (0/1/2/3+).
- Score: stm-perspective cp as stored by the eval worker (self-consistent with the
  working 2400-Elo pipeline).
- Format: nodchip 40-byte <32shHHbB> (unsigned move field).

Usage: pack_expert_bins.py <segments.jsonl> <outdir>
"""
import sys
import os
import json
import struct
import time
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~") + "/extnvme/phase-moe")
import chess

RESIDUE = {
    tuple(sorted(("".join(sorted(a)), "".join(sorted(b))))): name
    for (a, b), name in {
        # merged (rulings): rook-vs-2-minors in any form; queen-vs-material
        ("N", "B"): "nvb", ("R", "N"): "nvr", ("R", "B"): "bvr",
        ("R", "NN"): "rv2m", ("R", "NB"): "rv2m", ("R", "BB"): "rv2m",
        ("Q", "RR"): "qvmat", ("Q", "RN"): "qvmat", ("Q", "RB"): "qvmat",
    }.items()
}

HUFF = {
    chess.PAWN: (0b0001, 4), chess.KNIGHT: (0b0011, 4),
    chess.BISHOP: (0b0101, 4), chess.ROOK: (0b0111, 4),
    chess.QUEEN: (0b1001, 4), None: (0b0000, 1),
}


class BitWriter:
    def __init__(self):
        self.data = bytearray(32)
        self.bit = 0

    def one(self, b):
        if b:
            self.data[self.bit // 8] |= 1 << (self.bit & 7)
        self.bit += 1

    def n(self, d, n):
        for i in range(n):
            self.one(d & (1 << i))


def pack_sfen(board):
    w = BitWriter()
    w.one(1 if board.turn == chess.BLACK else 0)
    w.n(board.king(chess.WHITE), 6)
    w.n(board.king(chess.BLACK), 6)
    for rank in range(7, -1, -1):
        for file in range(8):
            sq = rank * 8 + file
            pc = board.piece_at(sq)
            if pc and pc.piece_type == chess.KING:
                continue
            code, bits = HUFF[pc.piece_type if pc else None]
            w.n(code, bits)
            if pc:
                w.one(1 if pc.color == chess.BLACK else 0)
    w.one(bool(board.castling_rights & chess.BB_H1))
    w.one(bool(board.castling_rights & chess.BB_A1))
    w.one(bool(board.castling_rights & chess.BB_H8))
    w.one(bool(board.castling_rights & chess.BB_A8))
    ep = board.ep_square
    if ep is None:
        w.one(0)
    else:
        w.one(1)
        w.n(ep, 6)
    w.n(board.halfmove_clock & 0x3F, 6)
    w.n(board.fullmove_number & 0xFF, 8)
    return bytes(w.data)


def pack_move(mv):
    raw = mv.from_square << 10 | mv.to_square << 4
    if mv.promotion:
        raw |= {chess.KNIGHT: 1, chess.BISHOP: 2,
                chess.ROOK: 3, chess.QUEEN: 4}[mv.promotion]
    return raw


def assign(r):
    """Return (expert_name, in_exchanges).  None expert = counted only."""
    w, b = r["config_i"].split("v")
    cw = [w.count(c) for c in "QRBN"]
    cb = [b.count(c) for c in "QRBN"]
    res_w, res_b = [], []
    for i, pc in enumerate("QRBN"):
        d = cw[i] - cb[i]
        if d > 0:
            res_w += [pc] * d
        elif d < 0:
            res_b += [pc] * (-d)
    kres = tuple(sorted(("".join(sorted(res_w)), "".join(sorted(res_b)))))
    symmetric = not res_w and not res_b
    men = r["men"]

    if men <= 5:
        return "tb-region", 0
    if not symmetric:
        name = RESIDUE.get(kres)
        if name:
            return name, r["exch"]
    if symmetric and r.get("bishops_i") == "opp_bishops":
        return "oppb", r["exch"]
    if men <= 10:
        return "dvoretsky", r["exch"]
    # gambit: NO cascade branch — that expert trains exclusively on the
    # already-curated pool (tbpools/GAMBIT.jsonl); re-derivation deprecated
    lk = r["lock_c"] if r["lock_c"] < 3 else 3
    return "balanced_l%d" % lk, r["exch"]


def main():
    inp, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    counts = defaultdict(int)
    skipped = tb = same_band = 0
    t0 = time.time()
    files = {}

    def fout(name):
        if name not in files:
            files[name] = open(os.path.join(outdir, name + ".bin"), "wb")
        return files[name]

    with open(inp) as f:
        for line in f:
            r = json.loads(line)
            b0, b1 = r["traj"].split("_to_")
            if b0 == b1:          # same-band: no demonstrated improvement
                same_band += 1
                continue
            expert, exch = assign(r)
            if expert == "tb-region":
                tb += 1
                continue
            cp = max(-30000, min(30000, r["cp"]))
            try:
                board = chess.Board(r["fen"])
                mv = chess.Move.from_uci(r["played"])
                if mv not in board.legal_moves:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue
            rec = struct.pack("<32shHHbB", pack_sfen(board), cp,
                              pack_move(mv), board.fullmove_number, 0, 0)
            fout(expert).write(rec)
            counts[expert] += 1
            if exch:               # dual membership: also the exchanges expert
                fout("exchanges").write(rec)
                counts["exchanges"] += 1

    for fh in files.values():
        fh.close()
    print("PACKED per expert (%ds):" % (time.time() - t0))
    for e, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print("  %-14s %10s" % (e, format(c, ",")))
    print("skipped(illegal): %d, same-band excluded: %d, tb-region(counted only): %d"
          % (skipped, same_band, tb))
    print("EXPERT-PACK-COMPLETE")


if __name__ == "__main__":
    main()
