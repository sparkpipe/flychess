"""Pack curated OTB segments into .bin training format.
Filters out draw-to-draw and other no-change trajectories per operator.
"""
import sys
import os
import struct
import json

sys.path.insert(0, os.path.expanduser("~") + "/extnvme/phase-moe")
import chess

INPUT = "/home/spec/chess-lab/otb_segments.jsonl"
OUTPUT = "/home/spec/chess-lab/curated_training.bin"

# only include trajectories where a skill was demonstrated
INCLUDE = {
    "equalize_to_convert",   # pressing from balanced
    "collapse_to_equalize",  # defensive recovery
    "equalize_to_win",       # pushing from equal to won
    "equalize_to_press",     # from balanced to pressing
    "defend_to_equalize",    # fighting back from worse
    "convert_to_win",        # finishing
    "press_to_convert",      # pressing to winning
    "collapse_to_convert",   # comeback from bad to winning
    "defend_to_convert",     # from worse to winning
    "collapse_to_defend",    # from bad to fighting
    "collapse_to_press",     # from bad to pressing
    "defend_to_press",       # from worse to pressing
    "press_to_win",          # pressing to win
    "collapse_to_win",       # from bad to won
    "defend_to_win",         # from worse to won
}

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
    n_packed = 0
    n_skipped = 0
    n_filtered = 0
    with open(INPUT) as f, open(OUTPUT, "wb") as out:
        for line in f:
            r = json.loads(line)
            traj = r["traj"]
            if traj not in INCLUDE:
                n_filtered += 1
                continue

            fen = r["fen"]
            played = r["played"]
            cp = r["cp"]
            cp = max(-30000, min(30000, cp))

            try:
                board = chess.Board(fen)
                mv = chess.Move.from_uci(played)
                if mv not in board.legal_moves:
                    n_skipped += 1
                    continue
            except Exception:
                n_skipped += 1
                continue

            pos = pack_sfen(board)
            mv_raw = pack_move(mv)
            out.write(struct.pack("<32shHHbB", pos, cp, mv_raw,
                                  board.fullmove_number, 0, 0))
            n_packed += 1
            if n_packed % 1000000 == 0:
                print(f"packed {n_packed} ({n_filtered} filtered, "
                      f"{n_skipped} skipped)", flush=True)

    print(f"CURATED PACKED: {n_packed} positions -> {OUTPUT}",
          flush=True)
    print(f"  filtered (no-skill trajectories): {n_filtered}",
          flush=True)
    print(f"  skipped (illegal/bad): {n_skipped}", flush=True)
    print("CURATED-PACK-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
