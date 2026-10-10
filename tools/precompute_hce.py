"""Parallel precompute of HCE features + domain tags for head training.
Saves to .npz so the head trainer loads instantly instead of re-computing."""
import sys, os, glob, struct
import numpy as np
import chess
from multiprocessing import Pool
sys.path.insert(0, "/home/spec/chess-lab/tools")
from audit_packer import unpack_sfen
from score_experts import domain

R = "/mnt/cold-raid6/chess-audit"
CACHE = sys.argv[1] if len(sys.argv) > 1 else R + "/stack_cache"

def hce(board):
    f = []
    for pt, v in [(chess.PAWN,1),(chess.KNIGHT,3),(chess.BISHOP,3),(chess.ROOK,5),(chess.QUEEN,9)]:
        f.append(len(board.pieces(pt, chess.WHITE)) * v / 9.0)
        f.append(len(board.pieces(pt, chess.BLACK)) * v / 9.0)
    f.append(1.0 if board.turn == chess.WHITE else -1.0)
    f.append(len(board.piece_map()) / 32.0)
    f.append(board.fullmove_number / 100.0)
    f.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    f.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    f.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    f.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    f.append(1.0 if board.ep_square is not None else 0.0)
    for side in (chess.WHITE, chess.BLACK):
        pawns = board.pieces(chess.PAWN, side)
        files = [0] * 8
        for s in pawns:
            files[chess.square_file(s)] += 1
        f += [x / 2.0 for x in files]
        f.append(sum(1 for x in files if x >= 2) / 4.0)
    return f

def process_batch(args):
    fens, = args
    hces, doms = [], []
    DOMS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3","nvb",
            "nvr","bvr","rv2m","qvmat","oppb","dvoretsky","exchanges","none"]
    D2I = {d:i for i,d in enumerate(DOMS)}
    for fen in fens:
        b = chess.Board(fen)
        hces.append(hce(b))
        try:
            d = domain(b.fen())
        except Exception:
            d = "none"
        doms.append(D2I.get(d, D2I["none"]))
    return np.array(hces, dtype=np.float32), np.array(doms, dtype=np.int32)

for cf in sorted(glob.glob(CACHE + "/*.uint8.npy")):
    name = os.path.basename(cf)[:-10]
    out_npz = CACHE + "/" + name + ".hce.npz"
    if os.path.exists(out_npz):
        print("skip", name, flush=True)
        continue
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
    # parallel batches
    BSIZE = 5000
    batches = [(fens[i:i+BSIZE],) for i in range(0, n, BSIZE)]
    with Pool(20) as pool:
        results = pool.map(process_batch, batches)
    all_hce = np.concatenate([r[0] for r in results])
    all_dom = np.concatenate([r[1] for r in results])
    np.savez(out_npz, hce=all_hce, domain=all_dom)
    print("done %s: %d positions" % (name, n), flush=True)
print("HCE-PRECOMPUTE-COMPLETE")
