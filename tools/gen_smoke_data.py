"""SMOKE: generate training data in nodchip .bin format for nnue-pytorch.

Bit-exact PackedSfenValue packer matching the C++ SfenPacker:
  LSB-first bit stream, Huffman piece codes, castling, ep, rule50, ply.
40 bytes per entry. SF self-play at depth, score = cp (stm perspective).
"""
import sys
import os
import struct
import subprocess
import random

sys.path.insert(0, "/home/spec/chess-lab")
import chess
import chess.engine

SF = "/home/spec/Stockfish/src/stockfish"
OUT = "/home/spec/chess-lab/nnue_smoke_data.bin"
GAMES = int(os.environ.get("GAMES", "20"))
DEPTH = int(os.environ.get("DEPTH", "6"))

HUFF = {
    chess.PAWN: (0b0001, 4),
    chess.KNIGHT: (0b0011, 4),
    chess.BISHOP: (0b0101, 4),
    chess.ROOK: (0b0111, 4),
    chess.QUEEN: (0b1001, 4),
    None: (0b0000, 1),
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


def pack_move(mv, board):
    raw = mv.from_square << 10 | mv.to_square << 4
    if mv.promotion:
        raw |= {chess.KNIGHT: 1, chess.BISHOP: 2,
                chess.ROOK: 3, chess.QUEEN: 4}[mv.promotion]
    return raw


def main():
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 64})
    rng = random.Random(42)
    n_rows = 0
    with open(OUT, "wb") as f:
        for gi in range(GAMES):
            board = chess.Board()
            for _ in range(rng.randrange(4, 12)):
                if not list(board.legal_moves):
                    break
                board.push(rng.choice(list(board.legal_moves)))
            while not board.is_game_over() and board.ply() < 200:
                info = eng.analyse(board, chess.engine.Limit(depth=DEPTH))
                sc = info["score"].pov(board.turn).score()
                if sc is None:
                    break
                sc = max(-30000, min(30000, sc))
                mv = info["pv"][0]
                pos = pack_sfen(board)
                mv_raw = pack_move(mv, board)
                result = 0  # unknown during play; fill later if needed
                entry = struct.pack("<32shhHbB", pos, sc, mv_raw,
                                    board.ply(), result,
                                    0)
                f.write(entry)
                n_rows += 1
                board.push(mv)
            if gi % 5 == 0:
                print(f"game {gi}, rows={n_rows}", flush=True)
    eng.quit()
    print(f"TOTAL: {n_rows} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
