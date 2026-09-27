"""Per-expert census v3: residue-based confrontation matching (operator ruling:
imbalance persists in any material context, e.g. BxN mainlines), opp-bishops by
square-color tag, balanced sharded by locked-pawn amount."""
import json
import sys
from collections import defaultdict

# merged classes (operator rulings 2026-09-27):
#   2NvR + NBvR + 2BvR -> Rv2minors
#   2RvQ + RNvQ + RBvQ -> Qvmaterial
RESIDUE_RAW = {
    ("B", "N"): "NvB",
    ("R", "N"): "NvR",
    ("R", "B"): "BvR",
    ("R", "NN"): "Rv2minors",
    ("R", "NB"): "Rv2minors",
    ("R", "BB"): "Rv2minors",
    ("Q", "RR"): "Qvmaterial",
    ("Q", "RN"): "Qvmaterial",
    ("Q", "RB"): "Qvmaterial",
}
# canonicalize BOTH the pair order AND the inner string order (alphabetical),
# matching how res_w/res_b strings are built (sorted() per piece)
RESIDUE = {
    tuple(sorted(("".join(sorted(a)), "".join(sorted(b))))): v
    for (a, b), v in RESIDUE_RAW.items()
}

experts = defaultdict(lambda: [0, 0])
balanced_lock = defaultdict(int)
balanced_lock_qual = defaultdict(int)
lk = 0
n = 0
for line in open(sys.argv[1]):
    r = json.loads(line)
    n += 1
    w, b = r["config_i"].split("v")
    cw = [w.count(c) for c in "QRBN"]
    cb = [b.count(c) for c in "QRBN"]
    res_w = []
    res_b = []
    for i, pc in enumerate("QRBN"):
        d = cw[i] - cb[i]
        if d > 0:
            res_w += [pc] * d
        elif d < 0:
            res_b += [pc] * (-d)
    kres = tuple(sorted(["".join(sorted(res_w)), "".join(sorted(res_b))]))
    symmetric = not res_w and not res_b
    men = r["men"]
    b0, b1 = r["traj"].split("_to_")
    draw2draw = r["traj"] == "equalize_to_equalize"

    e = None
    if men <= 5:
        e = "TB-region-OTB"
    else:
        if symmetric:
            np_w = "".join(sorted(x for x in w if x != "P"))
            np_b = "".join(sorted(x for x in b if x != "P"))
            if r.get("bishops_i") == "opp_bishops":
                e = "oppBishops"
        else:
            e = RESIDUE.get(kres)  # None = unlisted asymmetry -> falls through
    if e is None:
        if men <= 10:
            e = "dvoretsky"
        elif r["exch"]:
            e = "pieceTrades-approx"
        # NOTE: no cascade-gambit branch. The gambit expert trains EXCLUSIVELY on
        # the already-curated gambit pool (tbpools/GAMBIT.jsonl, operator-mined).
        # Cascade re-derivation of gambit data is DEPRECATED (operator correction).
        else:
            e = "balanced"
            lk = r["lock_c"] if r["lock_c"] < 3 else 3
            balanced_lock[lk] += 1

    experts[e][0] += 1
    # qualifying population: ANY band crossing = demonstrated improvement by
    # SOMEONE (40->55 helps; 55->60 same-band does not). With wp in white
    # perspective, a crossing down = the black side improving. Result-agnostic.
    if b0 != b1:
        experts[e][1] += 1
        if e == "balanced":
            balanced_lock_qual[lk if lk < 3 else 3] += 1

print("total: %d positions (338K games = ~22%% of pool)" % n)
print("%-22s%12s%11s%12s" % ("expert", "positions", "trainable", "x4-proj"))
for e, (p, t) in sorted(experts.items(), key=lambda kv: -kv[1][0]):
    if e:
        print("%-22s%12s%11s%12s" % (e, format(p, ","), format(t, ","), format(4 * p, ",")))
print("\nbalanced by center-locked files (0,1,2,3+): total / qualifying / x4-qual-proj")
tot = sum(balanced_lock.values())
for k in sorted(balanced_lock):
    print("  lock_c=%d: %11s (%5.1f%%)  qual %9s  x4-qual %10s" % (
        k, format(balanced_lock[k], ","), 100.0 * balanced_lock[k] / tot,
        format(balanced_lock_qual[k], ","), format(4 * balanced_lock_qual[k], ",")))
