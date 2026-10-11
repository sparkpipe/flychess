"""7-BUCKET DATA SPLIT: partition training data by phase predicates
(the same conditions the phase-moe fork routes on).

Reads the concatenated .bin file, decodes each 40-byte entry,
classifies by predicate, writes 7 bucket .bin files.

Buckets (operator taxonomy):
  0: opening-normal    (ply < 20, no gambit signature)
  1: opening-gambit    (ply < 20, material imbalance >= 1 pawn)
  2: mid-positional    (ply >= 20, queens on, closed/semi pawn structure)
  3: mid-open          (ply >= 20, queens on, open pawn structure)
  4: mid-queenless     (ply >= 20, no queens)
  5: mid-tactics       (ply >= 20, queens on, high piece tension)
  6: endgame-dvoretsky (piece count <= 10)

Note: we approximate these from the packed .bin format. The packed
position contains piece placements, so we can count pieces and detect
queens. We use gamePly as a proxy for opening/middlegame split.
For pawn structure and tactics, we use piece count heuristics.
"""
import sys
import os
import struct
import time

SRC = os.environ.get("SRC",
                     "/home/spec/chess-lab/fleet_data_all.bin")
OUTDIR = os.environ.get("OUTDIR", "/home/spec/chess-lab/buckets")
os.makedirs(OUTDIR, exist_ok=True)

# decode the packed sfen to count pieces and detect queens
PIECE_BITS = {0b0001: "P", 0b0011: "N", 0b0101: "B",
              0b0111: "R", 0b1001: "Q"}


def decode_pieces(data):
    """Extract piece info from 32-byte packed sfen."""
    pieces = []
    bit = 13  # after side(1) + wk(6) + bk(6)
    for sq in range(64):  # rank 8 down to 1, file a to h
        # check for piece
        b0 = (data[bit // 8] >> (bit & 7)) & 1
        if b0 == 0:
            bit += 1
            continue
        # read 4-bit code
        code = 0
        for i in range(4):
            bit_pos = bit + i
            code |= ((data[bit_pos // 8] >> (bit_pos & 7)) & 1) << i
        bit += 4
        # color bit
        color = (data[bit // 8] >> (bit & 7)) & 1
        bit += 1
        pt = PIECE_BITS.get(code, "?")
        pieces.append((sq, pt, color))
    return pieces, bit


def classify(pos_data, game_ply, score):
    """Return bucket index 0-6."""
    pieces, _ = decode_pieces(pos_data)
    n_pieces = len(pieces)
    has_queen = any(p[1] == "Q" for p in pieces)
    # count pawns for structure heuristic
    n_pawns = sum(1 for p in pieces if p[1] == "P")
    # material imbalance proxy from score magnitude
    imbalance = abs(score)

    if n_pieces <= 10:
        return 6  # endgame-dvoretsky
    if game_ply < 20:
        if imbalance >= 100:  # >= 1 pawn equivalent
            return 1  # opening-gambit
        return 0  # opening-normal
    # middlegame
    if not has_queen:
        return 4  # mid-queenless
    if n_pawns <= 6:  # open position (few pawns)
        return 3  # mid-open
    if imbalance >= 200:  # high tension
        return 5  # mid-tactics
    return 2  # mid-positional


def main():
    t0 = time.time()
    size = os.path.getsize(SRC)
    n_total = size // 40
    print(f"splitting {n_total} positions from {SRC}", flush=True)

    buckets = [open(f"{OUTDIR}/bucket_{i}.bin", "wb")
               for i in range(7)]
    counts = [0] * 7

    with open(SRC, "rb") as f:
        for i in range(n_total):
            entry = f.read(40)
            if len(entry) < 40:
                break
            pos_data = entry[:32]
            score = struct.unpack("<h", entry[32:34])[0]
            game_ply = struct.unpack("<H", entry[36:38])[0]
            b = classify(pos_data, game_ply, score)
            buckets[b].write(entry)
            counts[b] += 1
            if (i + 1) % 5000000 == 0:
                print(f"{i+1}/{n_total} "
                      f"({(i+1)/(time.time()-t0):.0f}/s)", flush=True)

    for b in buckets:
        b.close()
    for i, c in enumerate(counts):
        names = ["opening-norm", "opening-gambit", "mid-pos",
                 "mid-open", "mid-queenless", "mid-tactics",
                 "endgame-dvoretsky"]
        print(f"  bucket {i} ({names[i]}): {c} positions", flush=True)
    print(f"BUCKET-SPLIT-COMPLETE: {sum(counts)} total, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
