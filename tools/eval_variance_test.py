"""Quantify depth-12 eval variance: same positions, different engines/configs.
Configs: A x86 hash256, B x86 hash128 (cold engine per batch), C spark ARM hash128.
Reports |delta-cp| distribution and wall-flip rate (30/45/55/70 wp walls).
Usage (rtx5090): eval_variance_test.py <positions.txt> [n]
"""
import sys, os, math, random, subprocess, chess, chess.engine
from collections import defaultdict

SF = "/home/spec/Stockfish/src/stockfish"
POS = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
random.seed(42)

lines = [l.split("|")[0] for l in open(POS) if "|" in l]
sample = random.sample(lines, min(N, len(lines)))

def run(configs, fens):
    engs = {}
    for name, (binp, h) in configs.items():
        engs[name] = chess.engine.SimpleEngine.popen_uci(binp)
        engs[name].configure({"Threads": 1, "Hash": h})
    out = {name: [] for name in configs}
    for fen in fens:
        b = chess.Board(fen)
        for name in configs:
            info = engs[name].analyse(b, chess.engine.Limit(depth=12))
            out[name].append(info["score"].pov(b.turn).score())
    for e in engs.values():
        e.quit()
    return out

def stats(d):
    d = sorted(d)
    n = len(d)
    return ("med %4d  p90 %4d  max %4d  >30cp %4.1f%%"
            % (d[n // 2], d[int(n * 0.9)], d[-1], 100.0 * sum(x > 30 for x in d) / n))

def wall(wp):
    return 0.30 <= wp < 0.45 and "d" or 0.45 <= wp < 0.55 and "e" or \
           0.55 <= wp < 0.70 and "p" or wp >= 0.70 and "c" or "x"

def wp_of(cp):
    return 1.0 / (1.0 + math.exp(-cp / 361.0))

cfgs = {"A_x86_h256": (SF, 256), "B_x86_h128": (SF, 128)}
res = run(cfgs, sample)
pairs = [(a, b) for a, b in zip(res["A_x86_h256"], res["B_x86_h128"])
         if a is not None and b is not None]
print("   (skipped %d mate/stalemate scores)" % (len(sample) - len(pairs)))
dx = [abs(a - b) for a, b in pairs]
flips = sum(1 for a, b in pairs if wall(wp_of(a)) != wall(wp_of(b)))
print("hash256 vs hash128 (same binary):")
print("  ", stats(dx))
print("   wall flips: %d/%d (%.1f%%)" % (flips, len(pairs), 100.0 * flips / len(pairs)))
print("VARIANCE-TEST-DONE")
