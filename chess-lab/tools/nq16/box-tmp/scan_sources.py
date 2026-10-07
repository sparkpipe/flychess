import sys, os
from collections import Counter
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess

# 1) tb.tsv piece-count histogram
men = Counter()
for line in open("/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/tb.tsv", errors="ignore"):
    fen = line.split("|")[0]
    try:
        b = chess.Board(fen)
    except Exception:
        continue
    n = len(b.piece_map())
    men[n] += 1
print("tb.tsv piece-count histogram:", dict(sorted(men.items())))

# 2) complete asymmetric-residue enumeration over segment corpus
res = Counter()
SEG = "/extnvme/segments"
files = [f for f in sorted(os.listdir(SEG))
         if f.startswith("pos_") and (f.endswith(".clean.tsv") or f.endswith(".draws.tsv"))]
n = 0
for fn in files:
    for line in open(f"{SEG}/{fn}", errors="ignore"):
        p = line.rstrip("\n").split("|")
        if len(p) < 6 or p[3] != "pos":
            continue
        try:
            b = chess.Board(p[0])
        except Exception:
            continue
        wN = b.pieces_mask and (len(b.pieces(chess.KNIGHT, chess.WHITE)) - len(b.pieces(chess.KNIGHT, chess.BLACK)))
        d = (len(b.pieces(chess.KNIGHT, chess.WHITE)) - len(b.pieces(chess.KNIGHT, chess.BLACK)),
             len(b.pieces(chess.BISHOP, chess.WHITE)) - len(b.pieces(chess.BISHOP, chess.BLACK)),
             len(b.pieces(chess.ROOK, chess.WHITE)) - len(b.pieces(chess.ROOK, chess.BLACK)),
             len(b.pieces(chess.QUEEN, chess.WHITE)) - len(b.pieces(chess.QUEEN, chess.BLACK)))
        if any(d):
            res[d] += 1
        n += 1
        if n >= 4000000:
            break
    if n >= 4000000:
        break
print(f"\nscanned {n:,} positions; distinct nonzero piece-residues: {len(res)}")
named = {
 (0,1,-1,0): "BvR", (0,-1,1,0): "BvR(m)",
 (1,0,-1,0): "NvR", (-1,0,1,0): "NvR(m)",
 (0,0,-2,1): "QvRR", (0,0,2,-1): "QvRR(m)",
 (-1,0,-1,1): "QvRN", (1,0,1,-1): "QvRN(m)",
 (0,-1,-1,1): "QvRB", (0,1,1,-1): "QvRB(m)",
 (-2,0,1,0): "RvNN", (2,0,-1,0): "RvNN(m)",
 (0,-2,1,0): "RvBB", (0,2,-1,0): "RvBB(m)",
 (-1,-1,1,0): "RvNB", (1,1,-1,0): "RvNB(m)",
 (2,-2,0,0): "2Nv2B", (-2,2,0,0): "2Nv2B(m)",
 (1,-1,0,0): "NvB", (-1,1,0,0): "NvB(m)",
}
unnamed = Counter()
for d, c in res.most_common():
    if d not in named:
        unnamed[d] = c
print(f"UNNAMED residues (the catch-all territory): {len(unnamed)} distinct, "
      f"{sum(unnamed.values()):,} positions total")
for d, c in unnamed.most_common(25):
    print(f"  N{d[0]:+d} B{d[1]:+d} R{d[2]:+d} Q{d[3]:+d}: {c:,}")
