"""TRAJECTORY SEGMENT EXTRACTOR v2 — implements the operator rulings in full.

Differences from v1 (v1 was built on the partial dump and had 3 defects):
  1. Blunder boundary was compared in win-prob space (120/361=0.332 wp),
     which is a ~480cp swing near equality — NOT 120cp.  v2 compares
     |cp_i - cp_{i-1}| >= BLUNDER_CP directly in cp space.
  2. v1 tagged each segment with ONE config sampled at its middle position.
     RULING: sub-divide at every material change; each sub-segment trains
     its own specialist; the exchange move itself is tagged `exch`.
  3. v1 had no pawn-contact tagging.  RULING: open/closed is a spectrum of
     pawn contact — locked (facing pawns same file), tension (capturable
     pawn), open file (no pawns); edge files locked != closed; before any
     contact = neutral.  v2 emits raw counts per position; bucketization
     is decided from the full-data matrix, not hardcoded.

Trajectory bands (operator): press 55-70, convert 70-win, equalize 45-55,
defend 30-45, win 95+, collapse <30.  Same-band segments are emitted with
their traj name but EXCLUDED at packing (draw-to-draw ruling).
Games with any missing eval are dropped whole (v1 coerced None->0).

Input  (depth-12 fleet eval output, one line per winner position):
  fen|played|winner|ply|welo|belo|result|cp|wdl|best|depth
Output: one JSON per position, self-contained.
Usage: segment_extractor_v2.py < evals.txt > segments_v2.jsonl
"""
import sys
import os
import json
import math
import time
from collections import defaultdict
from multiprocessing import Pool

BLUNDER_CP = 120     # eval swing that starts a new segment (cp space, either side)
CP_MISSING_DROP = True

BANDS = [
    ("collapse", 0.00, 0.30),
    ("defend",   0.30, 0.45),
    ("equalize", 0.45, 0.55),
    ("press",    0.55, 0.70),
    ("convert",  0.70, 0.95),
    ("win",      0.95, 1.01),
]

CENTER_FILES = {2, 3, 4, 5}   # c d e f ; a b g h are edge files

# piece ordering for config strings: Q > R > B > N > P
ORDER = {"Q": 0, "R": 1, "B": 2, "N": 3, "P": 4}


def cp_to_winprob(cp):
    return 1.0 / (1.0 + math.exp(-cp / 361.0))


def band_of(wp):
    for name, lo, hi in BANDS:
        if lo <= wp < hi:
            return name
    return "win" if wp >= 0.95 else "collapse"


def parse_fen_pieces(fen):
    """Fast manual FEN placement parse -> dict square->symbol.
    Squares 0..63 with a1=0 (rank*8+file), like python-chess."""
    board = fen.split(" ")[0]
    pieces = {}
    sq = 56  # FEN starts at rank 8
    for ch in board:
        if ch == "/":
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            pieces[sq] = ch
            sq += 1
    return pieces


def position_tags(fen):
    """Config string, bishop colors, pawn-contact counts, total men."""
    pieces = parse_fen_pieces(fen)
    w, b = [], []
    wb_sq, bb_sq = [], []
    wp_files = [0] * 8
    bp_files = [0] * 8
    wp_sqs, bp_sqs = [], []
    men = 0
    for sq, ch in pieces.items():
        men += 1
        u = ch.upper()
        if u == "K":
            continue   # kings never appear in config strings; counted in men
        f, r = sq % 8, sq // 8
        if u == "P":
            if ch == "P":
                wp_files[f] += 1
                wp_sqs.append((f, r))
                w.append("P")
            else:
                bp_files[f] += 1
                bp_sqs.append((f, r))
                b.append("P")
            continue
        if u == "B":
            if ch == "B":
                wb_sq.append((f + r) % 2)
            else:
                bb_sq.append((f + r) % 2)
        (w if ch.isupper() else b).append(u)
    w.sort(key=lambda p: ORDER[p])
    b.sort(key=lambda p: ORDER[p])
    config = f"{''.join(w)}v{''.join(b)}"

    bishops = ""
    if wb_sq and bb_sq:
        # majority square-color of each side's bishops
        wbc = 1 if sum(wb_sq) * 2 > len(wb_sq) else 0
        bbc = 1 if sum(bb_sq) * 2 > len(bb_sq) else 0
        bishops = "opp_bishops" if wbc != bbc else "same_bishops"

    # contact: locked = facing pawns same file (white below black)
    lock_c = lock_e = 0
    for f in range(8):
        wr = [r for (pf, r) in wp_sqs if pf == f]
        br = [r for (pf, r) in bp_sqs if pf == f]
        if not wr or not br:
            continue
        # any white pawn directly below a black pawn on this file
        if min(br) - max(wr) == 1:
            if f in CENTER_FILES:
                lock_c += 1
            else:
                lock_e += 1
    # tension: a pawn of one side attacks a pawn of the other
    wp_set, bp_set = set(wp_sqs), set(bp_sqs)
    tension = 0
    for (f, r) in wp_sqs:
        for df in (-1, 1):
            if (f + df, r + 1) in bp_set:
                tension += 1
                break
    for (f, r) in bp_sqs:
        for df in (-1, 1):
            if (f + df, r - 1) in wp_set:
                tension += 1
                break
    open_files = sum(1 for f in range(8) if wp_files[f] == 0 and bp_files[f] == 0)

    return config, bishops, lock_c, lock_e, tension, open_files, men


def process_game(args):
    """Returns (list of json lines, per-game stats)."""
    gid, positions = args
    out = []
    n_seg = 0
    n_exch = 0
    if len(positions) < 2:
        return out, 0, 0

    cps = [p["cp"] for p in positions]
    wps = []
    for p in positions:
        wp = cp_to_winprob(p["cp"])
        if p["winner"] == "b":
            wp = 1.0 - wp
        wps.append(wp)

    tags = [position_tags(p["fen"]) for p in positions]
    configs = [t[0] for t in tags]

    # segment boundaries: blunder (cp swing) OR material change
    bounds = [0]
    for i in range(1, len(positions)):
        if abs(cps[i] - cps[i - 1]) >= BLUNDER_CP or configs[i] != configs[i - 1]:
            bounds.append(i)
    bounds.append(len(positions))

    for si in range(len(bounds) - 1):
        start, end = bounds[si], bounds[si + 1]
        s_wp, e_wp = wps[start], wps[end - 1]
        sb, eb = band_of(s_wp), band_of(e_wp)
        traj = f"{sb}_to_{eb}"
        n_seg += 1
        exch = 0
        if start > 0 and configs[start] != configs[start - 1]:
            exch = 1   # first position after a material change
            n_exch += 1
        cfg, bishops, lock_c, lock_e, tension, open_files, men = tags[start]
        seg = {
            "gid": gid,
            "traj": traj,
            "config": cfg,
            "prev_config": configs[start - 1] if start > 0 else "",
            "bishops": bishops,
            "seg_start_wp": round(s_wp, 4),
            "seg_end_wp": round(e_wp, 4),
            "seg_len": end - start,
        }
        for i in range(start, end):
            p = positions[i]
            ci, bi, lc, le, tn, of_, mn = tags[i]
            rec = dict(seg)
            rec.update({
                "fen": p["fen"],
                "played": p["played"],
                "winner": p["winner"],
                "ply": p["ply"],
                "cp": p["cp"],
                "wp": round(wps[i], 4),
                "config_i": ci,           # per-position config (sub-division law)
                "bishops_i": bi,
                "lock_c": lc, "lock_e": le, "tension": tn, "open": of_,
                "men": mn,
                "phase": ("tb" if mn <= 5 else "dvoretsky" if mn <= 10 else "midup"),
                "exch": 1 if (i == start and exch) else 0,
            })
            out.append(json.dumps(rec))
    return out, n_seg, n_exch


def main():
    t0 = time.time()
    n_in = n_games = n_seg_tot = n_exch_tot = n_dropped = 0
    current = []
    prev_ply = None
    gid = 0
    jobs = []
    pool = Pool(min(os.cpu_count() or 8, 24))

    def flush(done):
        nonlocal n_seg_tot, n_exch_tot
        for lines, ns, nx in done:
            sys.stdout.write("\n".join(lines))
            if lines:
                sys.stdout.write("\n")
            n_seg_tot += ns
            n_exch_tot += nx

    for line in sys.stdin:
        parts = line.rstrip("\n").split("|")
        if len(parts) < 8:
            continue
        if parts[7] == "" or parts[7] == "None":
            current.append(None)  # mark game unusable
            continue
        ply = int(parts[3])
        if current and (prev_ply is None or ply <= prev_ply):
            # game boundary: ply decrease, or first position after a gap
            if None not in current and len(current) >= 2:
                jobs.append((gid, current))
            elif None in current:
                n_dropped += 1
            gid += 1
            current = []
        current.append({
            "fen": parts[0], "played": parts[1], "winner": parts[2],
            "ply": ply, "cp": int(parts[7]),
        })
        prev_ply = ply
        n_in += 1
        if len(jobs) >= 500:
            flush(pool.map(process_game, jobs))
            n_games += len(jobs)
            jobs = []
        if n_in % 5_000_000 == 0:
            print(f"{n_in} pos, {n_games + len(jobs)} games, "
                  f"{n_seg_tot} segs, {n_exch_tot} exch, {time.time()-t0:.0f}s",
                  file=sys.stderr, flush=True)
    if None not in current and len(current) >= 2:
        jobs.append((gid, current))
    elif None in current:
        n_dropped += 1
    if jobs:
        flush(pool.map(process_game, jobs))
        n_games += len(jobs)
    pool.close()
    print(f"DONE {n_in} pos, {n_games} games, {n_seg_tot} segs, "
          f"{n_exch_tot} exch, {n_dropped} dropped(incomplete evals), "
          f"{time.time()-t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
