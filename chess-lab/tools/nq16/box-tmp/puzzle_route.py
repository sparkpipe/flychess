"""Route ALL puzzles.tsv positions under routerB -> per-expert puzzle counts."""
import sys
from collections import Counter
sys.path.insert(0, "/tmp")
from routerB import routeB

counts = Counter()
n = 0
for line in open("/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/puzzles.tsv", errors="ignore"):
    p = line.rstrip("\n").split("|")
    if not p or not p[0].strip():
        continue
    counts[routeB(p[0])] += 1
    n += 1
    if n % 500000 == 0:
        print(f"{n:,} routed", flush=True)
total = sum(counts.values())
print(f"TOTAL {total:,}")
ORDER = ["tb","mvr","rv2m","qvmat","nvb","piece_down","oppb","dv",
         "op_pawnimb","op_even_l0","op_even_l1","op_even_l2p",
         "mg_unsafe","mg_safe_sym","mg_safe_one_castled"]
for e in ORDER:
    print(f"{e:22s} {counts.get(e,0):9,d}")
