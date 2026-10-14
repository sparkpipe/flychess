#!/usr/bin/env python3
"""Material-balance breakdown of the training set, per routed expert.
Answers: (1) are piece-UP positions folded into piece_down/mvr/rv2m/qvmat?
(2) what % of unbalanced positions end up orphaned (no proper expert)?
Histograms: net material value bucket + top mismatch types per expert."""
import random
import sqlite3
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess
from routerB import routeB

DB = "/srv/workspace/chess-active/store/games.db"
SAMPLE = 2_000_000
random.seed(11)

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
fens = [r[0] for r in con.execute(
    "select fen from positions order by random() limit ?", (SAMPLE // 8,))]
# fill the rest from seg + nongame puzzles for a representative mix
fens += [r[0] for r in con.execute(
    "select fen from seg_positions order by random() limit ?", (SAMPLE // 8,))]
fens += [r[0] for r in con.execute(
    "select fen from nongame where source='puzzle' and label glob '[-0-9]*'"
    " order by random() limit ?", (SAMPLE // 8,))]
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

per_expert_val = defaultdict(Counter)     # expert -> net-value bucket -> n
per_expert_type = defaultdict(Counter)    # expert -> mismatch signature -> n
per_expert_side = defaultdict(Counter)    # expert -> stm-relative balance -> n
orphans = Counter()
unbalanced_total = 0
total = 0

for fen in fens:
    try:
        b = chess.Board(fen)
    except Exception:
        continue
    total += 1
    diffs = {}
    net = 0
    for pt, v in VAL.items():
        d = len(b.pieces(pt, chess.WHITE)) - len(b.pieces(pt, chess.BLACK))
        if d:
            diffs[pt] = d
            net += v * d
    if not diffs:
        continue  # material-even position (op_/mg/tb domains mostly)
    unbalanced_total += 1
    try:
        e = routeB(fen)
    except Exception:
        e = "ROUTE-FAIL"
    # stm-relative: is the side to move up or down material?
    stm_up = net > 0 if b.turn == chess.WHITE else net < 0
    if e == "ROUTE-FAIL":
        orphans["route-fail"] += 1
        continue
    # a proper imbalance expert is one of the named asym domains; anything else
    # receiving unbalanced positions = soft orphan (routed to a non-imbalance expert)
    IMBAL = {"mvr", "rv2m", "qvmat", "nvb", "piece_down"}
    if e not in IMBAL:
        orphans[f"routed-to-{e}"] += 1
    bucket = ("even" if net == 0 else
              f"{'+' if net > 0 else ''}{net // 2 * 2 if abs(net) >= 2 else net}"
              if abs(net) <= 9 else ("+10+" if net > 0 else "-10-"))
    per_expert_val[e][bucket] += 1
    sig = ",".join(f"{'W' if d > 0 else 'B'}{int(pt and 0)}".replace("0", "")
                   for pt, d in sorted(diffs.items(), key=lambda x: -VAL[x[0]]))
    sig = ",".join(f"{'+' if d > 0 else ''}{d}x{chess.piece_symbol(pt).upper()}"
                   for pt, d in sorted(diffs.items(), key=lambda x: -VAL[x[0]]))
    per_expert_type[e][sig] += 1
    per_expert_side[e]["stm_up" if stm_up else
                       ("stm_down" if net != 0 else "stm_even")] += 1

print(f"sampled {total:,} positions; unbalanced (any piece asym): "
      f"{unbalanced_total:,} ({100*unbalanced_total/total:.1f}%)")
o = sum(orphans.values())
print(f"ORPHANS (unbalanced but not in an imbalance expert): {o:,} "
      f"({100*o/max(unbalanced_total,1):.2f}% of unbalanced)")
for k, v in orphans.most_common(8):
    print(f"   {k}: {v:,}")
print()
for e in sorted(per_expert_val, key=lambda x: -sum(per_expert_val[x].values())):
    n = sum(per_expert_val[e].values())
    print(f"== {e}: {n:,} unbalanced positions")
    hist = per_expert_val[e]
    print("   net-value:", ", ".join(f"{k}:{v}" for k, v in
          sorted(hist.items(), key=lambda kv: float(kv[0].strip('+'))
                 if kv[0] not in ("even", "+10+", "-10-") else
                 (99 if kv[0] == "+10+" else -99))))
    print("   stm-relative:", dict(per_expert_side[e]))
    print("   top mismatch types:")
    for sig, c in per_expert_type[e].most_common(6):
        print(f"      {sig:<28} {c:>8,}  ({100*c/n:.1f}%)")
