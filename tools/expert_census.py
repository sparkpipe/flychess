"""Per-expert census v3: residue-based confrontation matching (operator ruling:
imbalance persists in any material context, e.g. BxN mainlines), opp-bishops by
square-color tag, balanced sharded by locked-pawn amount."""
import json
import sys
from collections import defaultdict

RESIDUE_RAW = {
    ("B", "N"): "NvB",
    ("NN", "NB"): "2NvNB",
    ("R", "N"): "NvR",
    ("R", "B"): "BvR",
    ("R", "NN"): "2NvR",
    ("R", "NB"): "NBvR",
    ("R", "BB"): "2BvR",
    ("Q", "RR"): "2RvQ",
    ("Q", "RN"): "RNvQ",
    ("Q", "RB"): "RBvQ",
}
RESIDUE = {tuple(sorted(k)): v for k, v in RESIDUE_RAW.items()}

experts = defaultdict(lambda: [0, 0])
balanced_lock = defaultdict(int)
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
    draw2draw = r["traj"] == "equalize_to_equalize"

    e = None
    if men <= 5:
        e = "TB-region-OTB"
    else:
        if symmetric:
            np_w = "".join(sorted(x for x in w if x != "P"))
            np_b = "".join(sorted(x for x in b if x != "P"))
            if np_w == "RR" and np_b == "RR":
                e = "2Rv2R"
            elif np_w == "BB" and np_b == "BB" and "N" not in w + b:
                e = "2Bv2B"
            elif r.get("bishops_i") == "opp_bishops":
                e = "oppBishops"
        else:
            e = RESIDUE.get(kres)  # None = unlisted asymmetry -> falls through
    if e is None:
        if men <= 10:
            e = "dvoretsky"
        elif r["exch"]:
            e = "pieceTrades-approx"
        elif r["ply"] < 24:
            e = "gambit-earlyasym"
        else:
            e = "balanced"
            balanced_lock[r["lock_c"] if r["lock_c"] < 3 else 3] += 1

    experts[e][0] += 1
    if not draw2draw:
        experts[e][1] += 1

print("total: %d positions (338K games = ~22%% of pool)" % n)
print("%-22s%12s%11s%12s" % ("expert", "positions", "trainable", "x4-proj"))
for e, (p, t) in sorted(experts.items(), key=lambda kv: -kv[1][0]):
    if e:
        print("%-22s%12s%11s%12s" % (e, format(p, ","), format(t, ","), format(4 * p, ",")))
print("\nbalanced by center-locked files (0,1,2,3+):")
tot = sum(balanced_lock.values())
for lk in sorted(balanced_lock):
    print("  lock_c=%d: %11s (%5.1f%%)" % (lk, format(balanced_lock[lk], ","), 100.0 * balanced_lock[lk] / tot))
