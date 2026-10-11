"""Per-expert training-material contribution + adjacent-ply expert switching.

Grouping: (source, gid, band) — rows within one band of one game are
ply-ordered and contiguous in the TSVs (wp segment design), so transitions
inside a group are real adjacent plies. Overlapping bands are counted
separately (contribution counts match assembly; switch counts sample each
band window once).
"""
import sys, os, json
from collections import Counter, defaultdict

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import router12 as R

SEG = "/extnvme/segments"
OUT = "/extnvme/active/contrib_main.json"

contrib = defaultdict(Counter)
switches = Counter()
same = 0
total_transitions = 0
groups = 0

rows_by_game = defaultdict(list)
for fn in sorted(os.listdir(SEG)):
    if not (fn.startswith("pos_") and (fn.endswith(".clean.tsv") or fn.endswith(".draws.tsv"))):
        continue
    src = "decisive" if fn.endswith(".clean.tsv") else "draws"
    band = fn[4:-10]
    for line in open(f"{SEG}/{fn}", errors="ignore"):
        p = line.rstrip("\n").split("|")
        if len(p) < 6:
            continue
        if p[3] != "pos":
            continue
        gid = p[5]
        if not gid.isdigit():
            continue
        rows_by_game[(src, gid, band)].append(p[0])

print(f"groups loaded: {len(rows_by_game)}", flush=True)

n = 0
for key, fens in rows_by_game.items():
    src = key[0]
    routed = R.route_seq(list(enumerate(fens)))
    prev = None
    for i, fen in enumerate(fens):
        e, role = routed[i]
        contrib[e][src] += 1
        if role == "AB":
            contrib[R.route23(fen)][src] += 1
        cur = R.route23(fen)
        if prev is not None:
            total_transitions += 1
            if cur != prev:
                switches[f"{prev}->{cur}"] += 1
            else:
                same += 1
        prev = cur
    groups += 1
    n += 1
    if n % 50000 == 0:
        print(f"processed {n} groups, {total_transitions} transitions", flush=True)

res = {
    "contrib": {e: dict(c) for e, c in contrib.items()},
    "switches": dict(switches.most_common(60)),
    "same": same,
    "total_transitions": total_transitions,
    "groups": groups,
}
json.dump(res, open(OUT, "w"), indent=1)
tot_sw = sum(switches.values())
print("DONE")
print(f"switch rate: {tot_sw}/{total_transitions} = "
      f"{100*tot_sw/max(1,total_transitions):.1f}% of adjacent plies change expert")
for k, v in switches.most_common(15):
    print(f"  {k}: {v}")
