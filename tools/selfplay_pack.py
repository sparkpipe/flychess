"""Pack self-play positions into per-expert training bins.
Every position classified by the cascade; cp is white-perspective from the
eval line (converted to stm-perspective for the .bin format)."""
import sys
import os
import struct

sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
from pack_expert_bins import pack_sfen, pack_move
from score_experts import domain

IN, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
files = {}
counts = {}
n = 0

for line in open(IN):
    parts = line.rstrip("\n").split("|")
    if len(parts) < 8:
        continue
    fen, mv_str, winner, ply = parts[0], parts[1], parts[2], int(parts[3])
    wcp = int(parts[7])   # white perspective
    board = chess.Board(fen)
    stm_black = fen.split(" ")[1] == "b"
    cp = -wcp if stm_black else wcp   # stm-perspective for .bin
    try:
        mv = chess.Move.from_uci(mv_str)
        if mv not in board.legal_moves:
            continue
    except Exception:
        continue
    d = domain(fen)
    if d in ("tb", "none"):
        continue
    rec = struct.pack("<32shHHbB", pack_sfen(board), max(-30000, min(30000, cp)),
                      pack_move(mv), board.fullmove_number, 0, 0)
    if d not in files:
        files[d] = open(os.path.join(OUT, d + ".bin"), "wb")
    files[d].write(rec)
    counts[d] = counts.get(d, 0) + 1
    n += 1

for fh in files.values():
    fh.close()
print("SELFPLAY PACK: %d positions" % n)
for d, c in sorted(counts.items(), key=lambda kv: -kv[1]):
    print("  %-14s %8s" % (d, format(c, ",")))
