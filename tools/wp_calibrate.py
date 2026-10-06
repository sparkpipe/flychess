"""WP CALIBRATION — measured from the depth-DB (both-side positions).

Reads all collected depth-DB shards (fen<TAB>cp@d1..cp@d20).
For each position: wp(cp@d20) vs actual game outcome.
The depth-DB covers positions from both sides of games (verified).
Game outcome is NOT in the shard — but the full position list came from
game bins where we know the result. We join back via the bin provenance.

Simpler: use the depth-DB evals (cp@d20) paired with the eval-fleet's
game results. The position lists overlap (same corpus).
"""
import sys, os, glob, math

R = "/mnt/cold-raid6/chess-audit"
DB = f"{R}/depth_db/sparks"

def wp(cp):
    return 1.0 / (1.0 + math.exp(-cp / 361.0))

# Build a lookup: fen-position-part -> result from the eval fleet
# The eval fleet has result per position (game result, not position result)
# Use field 6 (result) and field 0 (fen)
print("building result lookup from eval fleet...", flush=True)
result_map = {}
n_r = 0
SRC = "/mnt/cold-raid6/rtx5090-archive/chess-lab/otb_evals/combined.txt"
for line in open(SRC):
    parts = line.rstrip("\n").split("|")
    if len(parts) < 7:
        continue
    fen, result = parts[0], parts[6]
    key = fen.split(" ")[0]  # position part only
    if key not in result_map:
        result_map[key] = result
    n_r += 1
    if n_r >= 15736801:
        break
print(f"  {len(result_map):,} unique position-result pairs", flush=True)

# Now bin depth-DB positions by wp(cp@d20) -> observed outcome
NB = 20
bins = [[0, 0, 0] for _ in range(NB)]  # [count, mover_won, draws]
n_total = 0
n_matched = 0
for shard_f in sorted(glob.glob(f"{DB}/*/*.tsv.*")):
    for line in open(shard_f):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        fen = parts[0]
        cps = parts[1].split(",")
        if len(cps) < 20:
            continue
        try:
            cp20 = int(cps[19])  # depth 20, stm perspective
        except ValueError:
            continue
        n_total += 1
        key = fen.split(" ")[0]
        result = result_map.get(key)
        if result is None:
            continue
        n_matched += 1
        stm = "w" if " w " in fen else "b"
        stm_won = (result == "1-0" and stm == "w") or (result == "0-1" and stm == "b")
        drew = result == "1/2-1/2"
        p = wp(cp20)
        b = min(int(p * NB), NB - 1)
        bins[b][0] += 1
        if stm_won:
            bins[b][1] += 1
        if drew:
            bins[b][2] += 1

print(f"\nWP CALIBRATION from depth-DB ({n_matched:,} matched of {n_total:,} positions)")
print(f"pred_wp  obs_win  obs_draw  n")
print("-" * 45)
for i in range(NB):
    cnt, w, d = bins[i]
    if cnt < 100:
        continue
    pred = (i + 0.5) / NB
    obs = w / cnt
    print(f"   {pred:.2f}  {obs:.3f}  {d/cnt:.3f}  {cnt:>9,d}  {obs-pred:+.3f}")
