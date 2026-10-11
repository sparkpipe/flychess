#!/usr/bin/env python3
"""Label audit: decode train16 bin records, eval with SF17 d12, compare score."""
import random
import struct
import subprocess
import sys

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess
from pack_expert_bins import HUFF  # noqa: E402

BIN_DIR = "/srv/workspace/chess-active/train16"
SF17 = "/srv/workspace/flychess/src/sf17/src/stockfish"


class BitReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0  # absolute bit position

    def one(self):
        byte = self.data[self.pos // 8]
        v = (byte >> (7 - self.pos % 8)) & 1
        self.pos += 1
        return v

    def n(self, bits):
        v = 0
        for i in range(bits):
            v |= self.one() << i
        return v

    def many(self, bits):
        return self.n(bits)


# nodchip/SF training-format piece codes (LSB-first bit order, prefix-free):
# empty 0(1b); pawn 001(3b); knight 0101; bishop 0111; rook 1011; queen 1111 (4b)
def read_piece(r):
    if r.one() == 0:
        return None
    b2 = r.one()
    if b2 == 0:  # 0x01: pawn(3b) or knight(4b)
        b3 = r.one()
        if b3 == 0:
            return chess.PAWN
        r.one()
        return chess.KNIGHT
    b3 = r.one()
    if b3 == 0:  # 1,1,0,1 = rook
        r.one()
        return chess.ROOK
    b4 = r.one()
    if b4 == 0:  # 1,1,1,0 = bishop
        return chess.BISHOP
    return chess.QUEEN  # 1,1,1,1


def unpack_sfen(bits32):
    r = BitReader(bits32)
    stm = chess.BLACK if r.one() else chess.WHITE
    wk, bk = r.n(6), r.n(6)
    board = chess.Board(None)
    board.turn = stm
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    for rank in range(7, -1, -1):
        for file in range(8):
            sq = rank * 8 + file
            if sq in (wk, bk):
                continue
            pt = read_piece(r)
            if pt is None:
                continue
            color = chess.BLACK if r.one() else chess.WHITE
            board.set_piece_at(sq, chess.Piece(pt, color))
    cr = ""
    if r.one():
        cr += "K"
    if r.one():
        cr += "Q"
    if r.one():
        cr += "k"
    if r.one():
        cr += "q"
    board.set_castling_fen(cr or "-")
    if r.one():
        board.ep_square = r.n(6)
    else:
        board.ep_square = None
    hm = r.n(6)
    fm = r.n(8)
    board.halfmove_clock = hm
    board.fullmove_number = max(fm, 1)
    return board


def sample(expert, n=150, seed=7):
    random.seed(seed)
    recs = []
    path = f"{BIN_DIR}/{expert}.train.bin"
    size = 40
    total = os.path.getsize(path) // size
    idxs = sorted(random.sample(range(total), min(n, total)))
    with open(path, "rb") as f:
        prev = -1
        for i in idxs:
            f.seek(i * size)
            rec = f.read(size)
            _sfn, score = struct.unpack("<32sh", rec[:34])
            recs.append((score, rec[:32]))
    return recs


class SF:
    def __init__(self):
        self.p = subprocess.Popen([SF17], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.p.stdin.write("uci\nisready\n")
        self.p.stdin.flush()
        while "readyok" not in self.p.stdout.readline():
            pass

    def cp(self, fen, depth=12):
        self.p.stdin.write(f"position fen {fen}\ngo depth {depth}\n")
        self.p.stdin.flush()
        last = None
        while True:
            l = self.p.stdout.readline()
            if not l:
                return None
            if l.startswith("info ") and "score cp " in l and " pv " in l:
                t = l.split()
                for i, x in enumerate(t):
                    if x == "cp":
                        last = int(t[i + 1])
            elif l.startswith("bestmove"):
                return last


import os  # noqa: E402

sf = SF()
for expert in sys.argv[1:] or ["piece_down", "qvmat", "tb"]:
    rows = []
    for score, bits in sample(expert):
        try:
            b = unpack_sfen(bits)
            fen = b.fen()
            if not b.is_valid():
                raise ValueError("invalid board")
        except Exception:
            continue
        truth = sf.cp(fen)
        if truth is None:
            continue
        stm_white = "w" in fen.split()[1]
        truth_white = truth if stm_white else -truth
        rows.append((score, truth, truth_white, fen))
    n = len(rows)
    agree_stm = sum(1 for s, t, tw, f in rows if s * t > 0)
    agree_white = sum(1 for s, t, tw, f in rows if s * tw > 0)
    # scale regression (no intercept): packed = k * truth (same-pov hypothesis)
    num = sum(s * tw for s, t, tw, f in rows)
    den = sum(tw * tw for s, t, tw, f in rows)
    k = num / den if den else 0
    # correlation both ways
    import math
    def corr(key):
        xs = [r[{"s": 0, "t": 1, "w": 2}[key]] for r in rows]
        ys = [r[0] for r in rows]
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / n
        vx = sum((a - mx) ** 2 for a in xs) / n
        vy = sum((b - my) ** 2 for b in ys) / n
        return cov / math.sqrt(max(vx * vy, 1e-9))
    print(f"== {expert}: n={n}")
    print(f"   sign-match vs stm-pov truth: {agree_stm}/{n}   vs white-pov truth: {agree_white}/{n}")
    print(f"   corr(packed, stm)={corr('t'):+.3f}  corr(packed, white)={corr('w'):+.3f}")
    print(f"   scale k (packed/white-pov, OLS): {k:.3f}")
    for s, t, tw, f in rows[:6]:
        print(f"     packed={s:+6d} sf17stm={t:+6d} sf17white={tw:+6d}  {f[:45]}")
sf.p.stdin.write("quit\n")
