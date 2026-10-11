"""Fleet data generator: runs on each spark, one process per worker.
Writes timestamped .bin chunks. Self-terminates after TIMEOUT_MIN or
CHUNKS_PER_WORKER chunks. Idempotent resume via chunk numbering.
"""
import sys
import os
import struct
import subprocess
import random
import time
import glob

import os as _os
sys.path.insert(0, _os.path.expanduser("~") + "/extnvme/phase-moe")
import chess
import chess.engine

SF = _os.path.expanduser("~") + "/extnvme/phase-moe/sf/src/stockfish"
DATADIR = _os.path.expanduser("~") + "/extnvme/phase-moe/data"
DEPTH = int(os.environ.get("DEPTH", "10"))
GAMES = int(os.environ.get("GAMES", "500"))
TIMEOUT_MIN = int(os.environ.get("TIMEOUT_MIN", "60"))
WORKER_ID = os.environ.get("WORKER_ID", str(random.randint(0, 9999)))
NODE = os.environ.get("NODE", os.uname().nodename)

os.makedirs(DATADIR, exist_ok=True)

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
    t0 = time.time()
    timeout_s = TIMEOUT_MIN * 60
    chunk_num = 0
    total_rows = 0
    while time.time() - t0 < timeout_s:
        eng = chess.engine.SimpleEngine.popen_uci(SF)
        eng.configure({"Threads": 1, "Hash": 32})
        rng = random.Random(int(time.time()) + chunk_num * 7919)
        outfile = os.path.join(
            DATADIR, f"chunk_{NODE}_w{WORKER_ID}_c{chunk_num}.bin")
        rows = 0
        with open(outfile, "wb") as f:
            for gi in range(GAMES):
                if time.time() - t0 > timeout_s:
                    break
                board = chess.Board()
                for _ in range(rng.randrange(4, 12)):
                    if not list(board.legal_moves):
                        break
                    board.push(rng.choice(list(board.legal_moves)))
                while not board.is_game_over() and board.ply() < 200:
                    info = eng.analyse(board,
                                       chess.engine.Limit(depth=DEPTH))
                    sc = info["score"].pov(board.turn).score()
                    if sc is None:
                        break
                    sc = max(-30000, min(30000, sc))
                    mv = info["pv"][0]
                    pos = pack_sfen(board)
                    mv_raw = pack_move(mv)
                    f.write(struct.pack("<32shHHbB", pos, sc, mv_raw,
                                        board.ply(), 0, 0))
                    rows += 1
                    board.push(mv)
        eng.quit()
        total_rows += rows
        chunk_num += 1
        print(f"chunk {chunk_num}: {rows} rows "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"WORKER DONE: {total_rows} rows in {chunk_num} chunks",
          flush=True)


if __name__ == "__main__":
    main()
