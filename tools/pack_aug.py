"""Pack the gambit AUGMENTATION evals into per-expert aug_<expert>.bin files.
Every position classified by the standard cascade; no selection filter (the
whole winner arc trains); duplicates with main bins are deliberate.
Usage: pack_aug.py <gaug_evals_dir> <expert_bins_dir>
"""
import sys
import glob
import struct

sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
from pack_expert_bins import assign, pack_sfen, pack_move

indir, bindir = sys.argv[1], sys.argv[2]
out = {}
counts = {}
n = 0
for path in sorted(glob.glob(indir + "/w*.txt")):
    for line in open(path):
        p = line.rstrip("\n").split("|")
        if len(p) < 8 or p[7] in ("", "None"):
            continue
        try:
            b = chess.Board(p[0])
            mv = chess.Move.from_uci(p[1])
            if mv not in b.legal_moves:
                continue
            w, bl = [], []
            wbc, bbc = [], []
            for s, pc in b.piece_map().items():
                u = pc.symbol().upper()
                if u == "K":
                    continue
                (w if pc.color else bl).append(u)
                if u == "B":
                    (wbc if pc.color else bbc).append(
                        (chess.square_file(s) + chess.square_rank(s)) % 2)
            order = {"Q": 0, "R": 1, "B": 2, "N": 3, "P": 4}
            cw = "".join(sorted(w, key=lambda x: order[x]))
            cb = "".join(sorted(bl, key=lambda x: order[x]))
            rec = {"config_i": cw + "v" + cb,
                   "men": len(b.piece_map()), "exch": 0,
                   "ply": int(p[3]) if p[3].isdigit() else 1,
                   "lock_c": 0, "bishops_i": "", "cp": int(p[7]), "fen": p[0]}
            if wbc and bbc and \
                    (sum(wbc) * 2 > len(wbc)) != (sum(bbc) * 2 > len(bbc)):
                rec["bishops_i"] = "opp_bishops"
            lock = 0
            for f in range(8):
                wr = [chess.square_rank(s) for s in chess.SquareSet(
                    b.pieces(chess.PAWN, chess.WHITE))
                    if chess.square_file(s) == f]
                br = [chess.square_rank(s) for s in chess.SquareSet(
                    b.pieces(chess.PAWN, chess.BLACK))
                    if chess.square_file(s) == f]
                if wr and br and min(br) - max(wr) == 1 and 2 <= f <= 5:
                    lock += 1
            rec["lock_c"] = lock
            ex = assign(rec)
            if ex == "tb-region":
                continue
            blob = struct.pack("<32shHHbB", pack_sfen(b), rec["cp"],
                               pack_move(mv), b.fullmove_number, 0, 0)
            if ex not in out:
                out[ex] = open("%s/aug_%s.bin" % (bindir, ex), "wb")
            out[ex].write(blob)
            counts[ex] = counts.get(ex, 0) + 1
            n += 1
        except Exception:
            continue
for fh in out.values():
    fh.close()
print("AUG PACKED: %d positions" % n)
for e, c in sorted(counts.items(), key=lambda kv: -kv[1]):
    print("  %-14s %10s" % (e, format(c, ",")))
print("AUG-PACK-COMPLETE")
