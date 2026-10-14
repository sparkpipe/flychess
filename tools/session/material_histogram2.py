#!/usr/bin/env python3
"""Material histogram v2 — PIECE mismatches only (pawn differences ignored).
Orphan = piece-asymmetric position routed outside {tb, mvr, rv2m, qvmat, nvb,
piece_down}. Deep table of what piece_down (the asymmetry catch-all) holds."""
import random
import sqlite3
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess
from routerB import routeB

DB = "/srv/workspace/chess-active/store/games.db"
SAMPLE = 750_000
random.seed(11)

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
fens = [r[0] for r in con.execute(
    "select fen from positions order by random() limit ?", (SAMPLE // 3,))]
fens += [r[0] for r in con.execute(
    "select fen from seg_positions order by random() limit ?", (SAMPLE // 3,))]
fens += [r[0] for r in con.execute(
    "select fen from nongame where source='puzzle' and label glob '[-0-9]*'"
    " order by random() limit ?", (SAMPLE // 3,))]
VAL = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
IMBAL = {"mvr", "rv2m", "qvmat", "nvb", "piece_down"}
OWNED = IMBAL | {"tb"}  # tb owns <=5men by design, ahead of imbalance checks

per_val = defaultdict(Counter)
per_type = defaultdict(Counter)
per_side = defaultdict(Counter)
orphans = Counter()
orphan_types = Counter()
unbal = total = 0

for fen in fens:
    try:
        b = chess.Board(fen)
    except Exception:
        continue
    total += 1
    diffs = {}
    net = 0
    for pt, v in VAL.items():  # pawns ignored entirely
        d = len(b.pieces(pt, chess.WHITE)) - len(b.pieces(pt, chess.BLACK))
        if d:
            diffs[pt] = d
            net += v * d
    if not diffs:
        continue
    unbal += 1
    try:
        e = routeB(fen)
    except Exception:
        e = "ROUTE-FAIL"
    stm_up = net > 0 if b.turn == chess.WHITE else net < 0
    bucket = "even" if net == 0 else ("+10+" if net > 10 else
                                      ("-10-" if net < -10 else f"{net:+d}"))
    sig = ",".join(f"{d:+d}x{chess.piece_symbol(pt).upper()}"
                   for pt, d in sorted(diffs.items(), key=lambda x: -VAL[x[0]]))
    if e not in OWNED:
        orphans[e] += 1
        orphan_types[sig] += 1
        continue
    per_val[e][bucket] += 1
    per_type[e][sig] += 1
    per_side[e]["up" if stm_up else ("down" if net else "even")] += 1

print(f"sampled {total:,}; PIECE-mismatch positions (pawns ignored): "
      f"{unbal:,} ({100*unbal/total:.1f}%)")
o = sum(orphans.values())
print(f"ORPHANS (piece mismatch outside tb/imbalance experts): {o:,} "
      f"({100*o/max(unbal,1):.2f}% of piece-mismatched)")
for k, v in orphans.most_common(8):
    print(f"   {k}: {v:,}")
for sig, c in orphan_types.most_common(10):
    print(f"   orphan-type {sig}: {c:,}")
print()
for e in sorted(per_val, key=lambda x: -sum(per_val[x].values())):
    n = sum(per_val[e].values())
    top = 15 if e == "piece_down" else 8
    print(f"== {e}: {n:,} piece-mismatch positions")
    def sortkey(kv):
        if kv[0] == "even":
            return (0, 0.0)
        if kv[0] == "+10+":
            return (1, 99.0)
        if kv[0] == "-10-":
            return (1, -99.0)
        return (1, float(kv[0]))
    print("   net piece-value:", ", ".join(
        f"{k}:{v}" for k, v in sorted(per_val[e].items(), key=sortkey)))
    print(f"   stm: {dict(per_side[e])}")
    for sig, c in per_type[e].most_common(top):
        print(f"      {sig:<30} {c:>8,}  ({100*c/n:.1f}%)")
