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


# operator bands: defend 30-45, equalize 45-55, press 55-70, convert 70-win
GROUPS = [(0.0, "deep"), (0.30, "defend"), (0.45, "equalize"),
          (0.55, "press"), (0.70, "convert")]


def wp_group(wp):
    g = "loss"
    for lo, name in GROUPS:
        if wp >= lo:
            g = name
    return g


# the three categories (operator ruling): loss->equal, equal->strong, strong->win
# side-agnostic: white RISING pieces climb directly; white FALLING pieces are
# BLACK climbing (strong->equal fall = black loss->equal, etc.; win->strong
# fall = black loss->loss, which is not one of the three)
# the FOUR categories (operator ruling): 30->45, 45->60, 55->70, 70->win
# machine: pieces cut at band walls; closure on a qualifying climb of either
# side; convert pieces (>=70, no wall above) close at segment end while rising
CLIMB = {("defend", "equalize"): "30to45",
         ("equalize", "press"): "45to60",
         ("press", "convert"): "55to70",
         ("equalize", "defend"): "30to45",   # black climbing
         ("press", "equalize"): "45to60"}    # black climbing

experts = defaultdict(lambda: [0, 0])
cats = defaultdict(int)
cur_seg = None
piece_group = piece_expert = piece_lock = None
piece_len = piece_first_wp = None
balanced_lock = defaultdict(int)
balanced_lock_qual = defaultdict(int)
prev_group = None
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
    exch_first = r["exch"]
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
            e = "exchtrans"   # transient: not a piece-attribution class
        # NOTE: no cascade-gambit branch. The gambit expert trains EXCLUSIVELY on
        # the already-curated gambit pool (tbpools/GAMBIT.jsonl, operator-mined).
        # Cascade re-derivation of gambit data is DEPRECATED (operator correction).
        else:
            e = "balanced"
            lk = r["lock_c"] if r["lock_c"] < 3 else 3
            balanced_lock[lk] += 1

    experts[e][0] += 1

    # THREE-CATEGORY selection (operator ruling): loss->equal, equal->strong,
    # strong->win. A piece = run of positions inside one segment with constant
    # group; it is credited when it CLOSES on a qualifying rise (the climbing
    # side's step). Pieces interrupted by segment boundaries discard.
    seg_id = (r["gid"], r["traj"], r["seg_start_wp"], r["seg_end_wp"])
    g = wp_group(r["wp"])
    if seg_id != cur_seg or g != piece_group:
        if seg_id == cur_seg and CLIMB.get((piece_group, g)):
            rise = CLIMB[(piece_group, g)]
            experts[piece_expert][1] += piece_len
            cats[rise] += piece_len
            if piece_expert == "balanced":
                balanced_lock_qual[piece_lock] += piece_len
            if piece_exch:
                experts["exchanges"] = [experts["exchanges"][0],
                                        experts["exchanges"][1] + 1]
        elif seg_id != cur_seg and piece_group == "convert" \
                and piece_last_wp > piece_first_wp:
            experts[piece_expert][1] += piece_len   # 70->win (white converting)
            cats["70towin"] += piece_len
            if piece_exch:
                experts["exchanges"] = [experts["exchanges"][0],
                                        experts["exchanges"][1] + 1]
        elif seg_id != cur_seg and piece_group == "deep" \
                and piece_last_wp < piece_first_wp:
            experts[piece_expert][1] += piece_len   # 70->win (black converting)
            cats["70towin"] += piece_len
            if piece_exch:
                experts["exchanges"] = [experts["exchanges"][0],
                                        experts["exchanges"][1] + 1]
        cur_seg = seg_id
        piece_group = g
        piece_expert = "balanced" if e == "exchtrans" else e
        piece_lock = lk if lk < 3 else 3
        piece_exch = exch_first
        piece_len = 1
        piece_first_wp = piece_last_wp = r["wp"]
    else:
        piece_len += 1
    piece_last_wp = r["wp"]

print("total: %d positions (338K games = ~22%% of pool)" % n)
print("%-22s%12s%11s%12s" % ("expert", "positions", "trainable", "x4-proj"))
for e, (p, t) in sorted(experts.items(), key=lambda kv: -kv[1][0]):
    if e:
        print("%-22s%12s%11s%12s" % (e, format(p, ","), format(t, ","), format(4 * p, ",")))
print("\nby category (positions in qualifying pieces):")
for c, n in sorted(cats.items(), key=lambda kv: -kv[1]):
    print("  %-16s %11s" % (c, format(n, ",")))
print("\nbalanced by center-locked files (0,1,2,3+): total / qualifying / x4-qual-proj")
tot = sum(balanced_lock.values())
for k in sorted(balanced_lock):
    print("  lock_c=%d: %11s (%5.1f%%)  qual %9s  x4-qual %10s" % (
        k, format(balanced_lock[k], ","), 100.0 * balanced_lock[k] / tot,
        format(balanced_lock_qual[k], ","), format(4 * balanced_lock_qual[k], ",")))
