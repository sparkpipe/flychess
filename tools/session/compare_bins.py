#!/usr/bin/env python3
"""Compare the ORIGINAL piece_down bin labels (bins-fleet-v1, the files the
0.07-0.18 runs trained on) against MY relabeled d12 bin — per fen, in order.
Decodes the 32-byte huffman sfen with the inverse of pack_expert_bins."""
import struct
import sys
import chess

HUFF_DEC = {(0b0001, 4): chess.PAWN, (0b0011, 4): chess.KNIGHT,
            (0b0101, 4): chess.BISHOP, (0b0111, 4): chess.ROOK,
            (0b1001, 4): chess.QUEEN, (0b0, 1): None}
SYM = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
       chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}


class R:
    def __init__(self, data):
        self.d = data
        self.i = 0

    def one(self):
        b = (self.d[self.i // 8] >> (self.i & 7)) & 1
        self.i += 1
        return b

    def n(self, k):
        v = 0
        for j in range(k):
            v |= self.one() << j
        return v


def sq_name(sq):
    return chr(ord("a") + (sq % 8)) + str(sq // 8 + 1)


def unpack_sfen(b):
    r = R(b)
    stm = "b" if r.one() else "w"
    wk, bk = r.n(6), r.n(6)
    board = [""] * 64
    board[wk] = "K"
    board[bk] = "k"
    for rank in range(7, -1, -1):
        for file in range(8):
            sq = rank * 8 + file
            if sq in (wk, bk):
                continue
            code, bits = r.one(), 1
            if code:
                code3 = (r.one() << 1) | (r.one() << 0) | 0b1000 if False else 0
                # 4-bit code: 1xyz — read three more bits
                b2, b3, b4 = r.one(), r.one(), r.one()
                val = 0b1000 | (b2 << 2) | (b3 << 1) | b4
                pt = {0b0001: chess.PAWN, 0b0011: chess.KNIGHT,
                      0b0101: chess.BISHOP, 0b0111: chess.ROOK,
                      0b1001: chess.QUEEN}.get(val)
                if pt is None:
                    return None
                black = r.one()
                board[sq] = SYM[pt].lower() if black else SYM[pt]
    cast = "".join(c for c, f in zip("KQkq", (r.one(), r.one(), r.one(), r.one())) if f) or "-"
    ep = "-"
    if r.one():
        ep = sq_name(r.n(6))
    hm = r.n(6)
    fm = r.n(8)
    rows = []
    for rank in range(7, -1, -1):
        e = 0
        row = ""
        for file in range(8):
            pc = board[rank * 8 + file]
            if pc:
                if e:
                    row += str(e)
                    e = 0
                row += pc
            else:
                e += 1
        if e:
            row += str(e)
        rows.append(row)
    return f"{'/'.join(rows)} {stm} {cast} {ep} {hm} {fm}"


def read_bin(path, limit=None):
    out = []
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 40
    if limit:
        n = min(n, limit)
    for i in range(n):
        sfen, score, mv, ply, b, u = struct.unpack_from("<32shHHbB", raw, i * 40)
        out.append((sfen, score))
    return out


def main():
    orig = read_bin(sys.argv[1], limit=int(sys.argv[3]) if len(sys.argv) > 3 else None)
    mine = read_bin(sys.argv[2], limit=len(orig))
    print(f"records: orig={len(orig):,} mine={len(mine):,}")
    fen_mismatch = sign_flip = 0
    deltas = []
    big = []
    for i, ((os_, oc), (ms_, mc)) in enumerate(zip(orig, mine)):
        if i < 200 or i % 20000 == 0:   # spot-check fen alignment
            of, mf = unpack_sfen(os_), unpack_sfen(ms_)
            if of != mf:
                fen_mismatch += 1
                if fen_mismatch < 3:
                    print(f"FEN MISMATCH @{i}: {of} vs {mf}")
        if oc * mc < 0:
            sign_flip += 1
            if len(big) < 5:
                big.append((i, oc, mc))
        deltas.append(abs(oc - mc))
    deltas.sort()
    n = len(deltas)
    print(f"fen order mismatches (sampled): {fen_mismatch}")
    print(f"sign flips: {sign_flip:,} ({100*sign_flip/n:.2f}%)")
    print(f"|delta|: mean {sum(deltas)/n:.1f}  median {deltas[n//2]}  "
          f"p90 {deltas[int(n*0.9)]}  p99 {deltas[int(n*0.99)]}  max {deltas[-1]}")
    print(f"within 5cp: {100*sum(1 for d in deltas if d<=5)/n:.1f}%  "
          f"within 20cp: {100*sum(1 for d in deltas if d<=20)/n:.1f}%  "
          f">100cp: {100*sum(1 for d in deltas if d>100)/n:.2f}%")
    for i, oc, mc in big:
        print(f"  flip @{i}: orig {oc} vs mine {mc}")


if __name__ == "__main__":
    main()
