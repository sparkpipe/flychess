"""OTB → .BIN PACKER: converts evaluated OTB positions into the
nodchip .bin training format for nnue-pytorch.

Input: the deep-eval output file (fen|played|winner|ply|welo|belo|
result|cp|wdl|best|depth)
Output: .bin file ready for the trainer, with the played move as
label and the deep eval score as the target.

Can optionally filter by specialist predicate before packing.
"""
import sys
import os
import struct
import random

sys.path.insert(0, os.path.expanduser("~") + "/extnvme/phase-moe")
import chess

INPUT = os.environ.get("INPUT",
                       os.path.expanduser("~") +
                       "/extnvme/phase-moe/otb_evals.txt")
OUTPUT = os.environ.get("OUTPUT", "/home/spec/chess-lab/otb_training.bin")
MAX_ROWS = int(os.environ.get("MAX_ROWS", "0"))  # 0 = all

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


def pack_move(mv):
    raw = mv.from_square << 10 | mv.to_square << 4
    if mv.promotion:
        raw |= {chess.KNIGHT: 1, chess.BISHOP: 2,
                chess.ROOK: 3, chess.QUEEN: 4}[mv.promotion]
    return raw


def main():
    n = 0
    skipped = 0
    with open(INPUT) as f, open(OUTPUT, "wb") as out:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 11:
                skipped += 1
                continue
            fen = parts[0]
            played = parts[1]
            score = int(parts[7])  # centipawns from deep eval

            try:
                board = chess.Board(fen)
                mv = chess.Move.from_uci(played)
            except Exception:
                skipped += 1
                continue

            if mv not in board.legal_moves:
                skipped += 1
                continue

            score = max(-30000, min(30000, score))
            pos = pack_sfen(board)
            mv_raw = pack_move(mv)
            out.write(struct.pack("<32shHHbB", pos, score, mv_raw,
                                  board.fullmove_number, 0, 0))
            n += 1
            if MAX_ROWS and n >= MAX_ROWS:
                break
            if n % 1000000 == 0:
                print(f"{n} packed ({skipped} skipped)", flush=True)

    print(f"PACKED: {n} positions -> {OUTPUT} "
          f"({skipped} skipped)", flush=True)
    print("OTB-BIN-PACK-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
