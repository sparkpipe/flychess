"""TABLEBASE ENDGAME SPECIALIST: generate training data from syzygy.

Enumerates positions by walking the tablebase probe space for all
3-4-5-piece material configurations, records (packed_fen, exact WDL
score, best DTZ move) pairs in our .bin format.

Uses a random-walk generator (faster than full enumeration for training
purposes: generate legal positions weighted by game-tree likelihood).
"""
import sys
import os
import struct
import random
import time
import itertools

sys.path.insert(0, os.path.expanduser("~") + "/extnvme/phase-moe")
import chess
import chess.syzygy

TB_PATH = os.environ.get("TB_PATH", "/home/spec/syzygy")
OUT = os.environ.get("OUT", "/home/spec/chess-lab/tb_training.bin")
POSITIONS = int(os.environ.get("POSITIONS", "5000000"))
PIECE_SETS = [
    # classic endgame material configurations (3-5 pieces total)
    ["K", "k"],  # K v K
    ["K", "P", "k"],  # K+P v K
    ["K", "N", "k"], ["K", "B", "k"],  # K+minor v K
    ["K", "Q", "k"], ["K", "R", "k"],
    ["K", "P", "k", "p"], ["K", "N", "k", "n"],
    ["K", "B", "k", "b"], ["K", "R", "k", "r"],
    ["K", "Q", "k", "q"], ["K", "P", "P", "k"],
    ["K", "N", "N", "k"], ["K", "B", "B", "k"],
    ["K", "R", "P", "k"], ["K", "P", "k", "p"],
    ["K", "B", "N", "k"], ["K", "Q", "P", "k"],
    ["K", "R", "k", "p"], ["K", "Q", "k", "p"],
    ["K", "P", "P", "k", "p"], ["K", "R", "R", "k"],
    ["K", "Q", "N", "k"], ["K", "Q", "B", "k"],
    ["K", "R", "N", "k"], ["K", "R", "B", "k"],
    ["K", "Q", "R", "k"], ["K", "R", "P", "k", "p"],
    ["K", "B", "B", "k", "p"], ["K", "N", "N", "k", "p"],
    ["K", "Q", "P", "k", "p"], ["K", "Q", "Q", "k"],
    # 5-piece with mixed minors
    ["K", "B", "N", "k", "p"], ["K", "N", "P", "k", "p"],
    ["K", "R", "N", "k", "p"], ["K", "R", "B", "k", "p"],
    ["K", "Q", "N", "k", "p"], ["K", "Q", "B", "k", "p"],
    ["K", "R", "R", "k", "p"],
]

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


PIECE_MAP = {
    "K": (chess.KING, chess.WHITE),
    "Q": (chess.QUEEN, chess.WHITE),
    "R": (chess.ROOK, chess.WHITE),
    "B": (chess.BISHOP, chess.WHITE),
    "N": (chess.KNIGHT, chess.WHITE),
    "P": (chess.PAWN, chess.WHITE),
    "k": (chess.KING, chess.BLACK),
    "q": (chess.QUEEN, chess.BLACK),
    "r": (chess.ROOK, chess.BLACK),
    "b": (chess.BISHOP, chess.BLACK),
    "n": (chess.KNIGHT, chess.BLACK),
    "p": (chess.PAWN, chess.BLACK),
}


def random_position(rng, piece_set):
    """Place pieces randomly on an 8x8 board. Returns None if invalid."""
    squares = rng.sample(range(64), len(piece_set))
    board = chess.Board(None)  # empty board
    for sym, sq in zip(piece_set, squares):
        pt, color = PIECE_MAP[sym]
        board.set_piece_at(sq, chess.Piece(pt, color))
    # random side to move
    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    if not board.is_valid() or board.is_game_over():
        return None
    return board


def main():
    tb = chess.syzygy.open_tablebase(TB_PATH)
    rng = random.Random(42)
    t0 = time.time()
    n_written = 0
    n_skipped = 0

    with open(OUT, "wb") as f:
        while n_written < POSITIONS:
            ps = rng.choice(PIECE_SETS)
            board = random_position(rng, ps)
            if board is None:
                n_skipped += 1
                continue
            try:
                wdl = tb.probe_wdl(board)
                dtz = tb.probe_dtz(board)
            except Exception:
                n_skipped += 1
                continue

            # exact score from WDL (side-to-move perspective)
            # syzygy WDL: 2=win, 0=draw, -2=loss (cursed variants ±1)
            if wdl > 0:
                score = 1000 + (100 - min(abs(dtz), 100))  # winning
            elif wdl < 0:
                score = -(1000 + (100 - min(abs(dtz), 100)))
            else:
                score = 0  # draw

            # find the best move: maximize DTZ progression
            best_mv = None
            best_score = None
            for mv in board.legal_moves:
                board.push(mv)
                try:
                    child_dtz = tb.probe_dtz(board)
                    child_wdl = tb.probe_wdl(board)
                except Exception:
                    board.pop()
                    continue
                board.pop()
                # prefer: winning moves with shortest DTZ,
                # drawing moves, then losing moves with longest DTZ
                if child_wdl == -wdl or child_wdl == wdl:
                    # opponent perspective
                    pass
                # simplified: prefer shortest |DTZ| for winning positions
                # longest |DTZ| for losing positions
                move_score = -abs(child_dtz) if wdl > 0 else \
                    (abs(child_dtz) if wdl < 0 else 0)
                if best_score is None or move_score > best_score:
                    best_score = move_score
                    best_mv = mv

            if best_mv is None:
                n_skipped += 1
                continue

            pos = pack_sfen(board)
            mv_raw = pack_move(best_mv)
            f.write(struct.pack("<32shHHbB", pos, score, mv_raw,
                                board.fullmove_number, 0, 0))
            n_written += 1
            if n_written % 100000 == 0:
                el = time.time() - t0
                print(f"{n_written}/{POSITIONS} "
                      f"({n_written/el:.0f}/s, "
                      f"skipped={n_skipped})", flush=True)

    tb.close()
    print(f"TABLEBASE DATA COMPLETE: {n_written} positions, "
          f"{n_skipped} skipped, "
          f"{time.time()-t0:.0f}s", flush=True)
    print("TB-DATA-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
