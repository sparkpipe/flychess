"""Precompute HCE+domain for ONE bin (called from shell parallel loop)."""
import sys, os, struct
import numpy as np
import chess
sys.path.insert(0, "/home/spec/chess-lab/tools")
from audit_packer import unpack_sfen
from score_experts import domain

R = "/mnt/cold-raid6/chess-audit"
name = sys.argv[1]
CACHE = sys.argv[2] if len(sys.argv) > 2 else R + "/stack_cache"
cf = CACHE + "/" + name + ".uint8.npy"
out_npz = CACHE + "/" + name + ".hce.npz"
if os.path.exists(out_npz):
    print("skip", name)
    sys.exit(0)

def hce(board):
    f = []
    for pt, v in [(chess.PAWN,1),(chess.KNIGHT,3),(chess.BISHOP,3),(chess.ROOK,5),(chess.QUEEN,9)]:
        f.append(len(board.pieces(pt, chess.WHITE)) * v / 9.0)
        f.append(len(board.pieces(pt, chess.BLACK)) * v / 9.0)
    f.append(1.0 if board.turn == chess.WHITE else -1.0)
    f.append(len(board.piece_map()) / 32.0)
    f.append(board.fullmove_number / 100.0)
    for cr in [board.has_kingside_castling_rights(chess.WHITE),
               board.has_queenside_castling_rights(chess.WHITE),
               board.has_kingside_castling_rights(chess.BLACK),
               board.has_queenside_castling_rights(chess.BLACK),
               board.ep_square is not None]:
        f.append(1.0 if cr else 0.0)
    for side in (chess.WHITE, chess.BLACK):
        pawns = board.pieces(chess.PAWN, side)
        files = [0] * 8
        for s in pawns:
            files[chess.square_file(s)] += 1
        f += [x / 2.0 for x in files]
        f.append(sum(1 for x in files if x >= 2) / 4.0)
    return f

DOMS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3","nvb",
        "nvr","bvr","rv2m","qvmat","oppb","dvoretsky","exchanges","none"]
D2I = {d:i for i,d in enumerate(DOMS)}

src_bin = R + "/expert_bins_both/" + name + ".bin"
fens = []
with open(src_bin, "rb") as f:
    for i in range(os.path.getsize(src_bin) // 40):
        r = f.read(40)
        if len(r) < 40: break
        b, hm, fm = unpack_sfen(r[:32])
        fens.append(b.fen())
n = min(np.load(cf, mmap_mode="r").shape[0], len(fens))
fens = fens[:n]

hces = np.zeros((n, 36), dtype=np.float32)
doms = np.zeros(n, dtype=np.int32)
for i, fen in enumerate(fens):
    b = chess.Board(fen)
    hces[i] = hce(b)
    try:
        d = domain(b.fen())
    except Exception:
        d = "none"
    doms[i] = D2I.get(d, D2I["none"])
    if i % 500000 == 0 and i:
        print("  %s: %d/%d" % (name, i, n), flush=True)
np.savez(out_npz, hce=hces, domain=doms)
print("done %s: %d" % (name, n))
