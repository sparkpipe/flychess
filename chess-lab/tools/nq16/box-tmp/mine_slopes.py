"""MINE 2400+ SLOPE POSITIONS — operator-corrected filter.

From otb_complete_dump_both.txt (126M lines, fen|move|winner|ply|welo|belo|result):
  Filter: at least one player 2400+
  Extract: positions where a player's eval crosses one of the 4 bands
    (30→45, 45→60, 55→70, 70→win) AGAINST the 2400+ opponent.
  Sampling: winner's side (the climber), every other ply, stm-normalized.
Output: /mnt/cold-raid6/chess-audit/mined/climbing_2400.txt (fen|cp|welo|belo|side)
"""
import sys, os, re

R = "/mnt/cold-raid6/chess-audit"
SRC = f"{R}/otb_complete_both.txt" if os.path.exists(f"{R}/otb_complete_both.txt") else f"{R}/otb_complete_dump_both.txt"
OUT = f"{R}/mined/climbing_2400.txt"
ELO_FLOOR = 2400
BANDS = [(0.30, 0.45), (0.45, 0.60), (0.55, 0.70), (0.70, 1.01)]

# dump lines don't carry cp (the eval fleet output does). We need eval files.
# check: is there a matching eval file?
EVALS = f"{R}/otb_evals"
print(f"source: {SRC}")
print(f"evals dir exists: {os.path.exists(EVALS)}")
if os.path.exists(EVALS):
    files = sorted(os.listdir(EVALS))[:3]
    print(f"eval files: {files}")
    # sample a line
    with open(f"{EVALS}/{files[0]}") as f:
        line = f.readline().strip()
        parts = line.split("|")
        print(f"fields: {len(parts)}, sample: {parts[:4]}...")

# alternative: the dump lines have 7 fields: fen|move|winner|ply|welo|belo|result
# but no cp. The eval fleet output has: fen|move|winner|ply|welo|belo|result|cp|...
# check both the dump and the eval fleet
print(f"\ndump line fields:")
with open(SRC) as f:
    for i, line in enumerate(f):
        parts = line.strip().split("|")
        print(f"  {len(parts)} fields: {[p[:20] for p in parts]}")
        if i >= 2:
            break

# check for pre-computed eval fleet shards
import glob
shards = glob.glob(f"{R}/otb_evals/w*.txt") + glob.glob(f"{R}/otb_evals/*.txt")
print(f"\neval fleet shards: {len(shards)}")
if shards:
    with open(shards[0]) as f:
        line = f.readline().strip()
        parts = line.split("|")
        print(f"  fields ({len(parts)}): {[p[:15] for p in parts[:8]]}")
