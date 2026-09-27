import json
from collections import defaultdict, Counter

matrix_plys = defaultdict(int)
traj_totals = defaultdict(lambda: {"segs": 0, "plys": 0})
config_totals = defaultdict(lambda: {"segs": 0, "plys": 0})
segs_seen = set()

for line in open("/home/spec/chess-lab/otb_segments.jsonl"):
    r = json.loads(line)
    t = r["traj"]
    c = r["config"]
    matrix_plys[(t, c)] += 1
    traj_totals[t]["plys"] += 1
    config_totals[c]["plys"] += 1
    key = (r["gid"], r["traj"], r["seg_start_wp"])
    if key not in segs_seen:
        segs_seen.add(key)
        traj_totals[t]["segs"] += 1
        config_totals[c]["segs"] += 1

print("=== TRAJECTORY TOTALS ===")
print(f"{'trajectory':<28} {'segs':>8} {'positions':>10} {'avg_len':>8}")
for t in sorted(traj_totals, key=lambda x: -traj_totals[x]["plys"]):
    d = traj_totals[t]
    avg = d["plys"] / max(d["segs"], 1)
    print(f"{t:<28} {d['segs']:>8} {d['plys']:>10} {avg:>8.1f}")

total_segs = sum(d["segs"] for d in traj_totals.values())
total_plys = sum(d["plys"] for d in traj_totals.values())
print(f"\nTOTAL: {total_segs} segments, {total_plys} positions")

print()
print("=== TOP 30 MATERIAL CONFIGS ===")
print(f"{'config':<52} {'segs':>8} {'positions':>10}")
for c in sorted(config_totals, key=lambda x: -config_totals[x]["plys"])[:30]:
    d = config_totals[c]
    print(f"{c:<52} {d['segs']:>8} {d['plys']:>10}")

print(f"\nDISTINCT CONFIGS: {len(config_totals)}")

pc_counts = Counter()
for c, d in config_totals.items():
    pieces = c.split("v")
    total_np = sum(len(s.replace("P", "")) for s in pieces)
    pc_counts[total_np] += d["plys"]

print()
print("=== POSITIONS BY NON-PAWN PIECE COUNT ===")
for np_ in sorted(pc_counts):
    print(f"  {np_} non-pawn pieces: {pc_counts[np_]:>10} positions")

print()
print("=== TRAJECTORY × PIECE-COUNT MATRIX ===")
print("(positions)")
pc_by_traj = defaultdict(lambda: defaultdict(int))
for (t, c), plys in matrix_plys.items():
    pieces = c.split("v")
    np_ = sum(len(s.replace("P", "")) for s in pieces)
    pc_by_traj[t][np_] += plys

np_range = sorted(pc_counts)
top_trajs = sorted(traj_totals, key=lambda x: -traj_totals[x]["plys"])[:8]
header = "traj\\np      " + "".join(f"{n:>10}" for n in np_range)
print(header)
for t in top_trajs:
    row = f"{t:<20}"
    for n in np_range:
        v = pc_by_traj[t].get(n, 0)
        row += f"{v:>10}"
    print(row)
