"""Pack the tactics expert bin from the puzzle-combination evals.
Every position of every winning combination is correct (ruling) — no
selection filter; score = the depth-12 eval (stm perspective).
Usage: pack_tactics_bin.py <evals_dir_with_w*.txt> <out.bin>
"""
import glob
import sys
import chess

sys.path.insert(0, "/home/spec/chess-lab/tools")
from pack_expert_bins import pack_sfen, pack_move
import struct

indir, out = sys.argv[1], sys.argv[2]
n = skipped = 0
with open(out, "wb") as w:
    for path in sorted(glob.glob(indir + "/w*.txt")):
        for line in open(path):
            parts = line.rstrip("\n").split("|")
            if len(parts) < 8 or parts[7] in ("", "None"):
                continue
            try:
                b = chess.Board(parts[0])
                mv = chess.Move.from_uci(parts[1])
                if mv not in b.legal_moves:
                    skipped += 1
                    continue
                cp = max(-30000, min(30000, int(parts[7])))
                w.write(struct.pack("<32shHHbB", pack_sfen(b), cp,
                                    pack_move(mv), b.fullmove_number, 0, 0))
                n += 1
            except Exception:
                skipped += 1
print("TACTICS PACKED: %d positions (%d skipped)" % (n, skipped))
print("TACTICS-PACK-COMPLETE")
