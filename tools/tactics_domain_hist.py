"""Domain histogram of the tactics bin (23.3M records, full pass):
how many positions of each expert-domain live in the puzzle data."""
import sys, struct
sys.path.insert(0, "/home/spec/chess-lab/tools")
from audit_packer import unpack_sfen
from score_experts import domain
from collections import Counter

hist = Counter()
n = 0
with open("/mnt/cold-raid6/chess-audit/expert_bins_both/tactics.bin", "rb") as f:
    while True:
        r = f.read(40)
        if len(r) < 40:
            break
        b, hm, fm = unpack_sfen(r[:32])
        hist[domain(b.fen())] += 1
        n += 1
        if n % 2000000 == 0:
            print(n, dict(hist), flush=True)
for d, c in hist.most_common():
    print("%-14s %10s (%.2f%%)" % (d, format(c, ","), 100.0 * c / n))
print("TOTAL", n)
