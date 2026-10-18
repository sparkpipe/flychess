#!/usr/bin/env python3
"""Trial: how much of the AJ corpus survives the d10 training filter?

Applies the standard selection rules — verbatim — to the d10 trajectories
the labeling fleet is producing, to size the d20 follow-up labeling before
committing fleet time:

  segments (extractor v2): decisive games; winner-pov cp; segments split
  at blunder boundaries |dcp| >= 120; a segment's positions are KEPT iff
  its start band != end band; bands on wp = sigmoid(cp/361): collapse
  <0.30 / defend <0.45 / equalize <0.55 / press <0.70 / convert <0.95 / win

  miniatures (clean short wins): winner reaches >=200cp, never drops below
  200cp after, no single >=300cp jump before reaching 200cp, and game ends
  or winner >=500cp by ply 40 (move 20) — ENTIRE game kept.

AJ jsonl evals are white-pov per ply (worker converts stm->white).

Usage: aj_filter_trial.py --out DIR [--shards 000,001] [--max-games N]
"""
import argparse
import json
import math
import os

wp_of = lambda cp: 1.0 / (1.0 + math.exp(-cp / 361.0))


def band(wp):
    if wp < 0.30:
        return "collapse"
    if wp < 0.45:
        return "defend"
    if wp < 0.55:
        return "equalize"
    if wp < 0.70:
        return "press"
    if wp < 0.95:
        return "convert"
    return "win"


def seg_positions(wseq):
    """Indices of plies inside band-transition segments (extractor v2)."""
    segs = []
    cur = [0]
    for i in range(1, len(wseq)):
        if abs(wseq[i] - wseq[i - 1]) >= 120:
            segs.append(cur)
            cur = []
        cur.append(i)
    segs.append(cur)
    out = set()
    for seg in segs:
        if band(wp_of(wseq[seg[0]])) != band(wp_of(wseq[seg[-1]])):
            out.update(seg)
    return out


def is_clean_short_win(wseq):
    TWO_PAWNS, ROOK, BY_PLY, MAX_JUMP = 200, 500, 40, 300
    tp = next((i for i, v in enumerate(wseq) if v >= TWO_PAWNS), None)
    if tp is None or tp >= BY_PLY:
        return False
    pre = wseq[:tp + 1]
    if any(abs(b - a) >= MAX_JUMP for a, b in zip(pre, pre[1:])):
        return False
    if any(v < TWO_PAWNS for v in wseq[tp:]):
        return False
    return len(wseq) - 1 <= BY_PLY or any(v >= ROOK for v in wseq[:BY_PLY + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="aj-kept/out dir with jsonl")
    ap.add_argument("--shards", default="000,001")
    ap.add_argument("--max-games", type=int, default=0)
    a = ap.parse_args()

    games = read = draws = kept_seg_games = kept_seg_pos = 0
    kept_mini_games = kept_mini_pos = union_pos = 0
    band_counts = {}
    plies_total = 0

    bad = 0
    for sh in a.shards.split(","):
        path = os.path.join(a.out, f"aj.{sh}.jsonl")
        for line in open(path):
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            read += 1
            if a.max_games and read > a.max_games:
                break
            r = g.get("r")
            if r not in ("1-0", "0-1"):
                draws += 1
                continue
            wseq = [v if r == "1-0" else -v for v in g["e"]]
            games += 1
            plies_total += len(wseq)
            seg = seg_positions(wseq)
            if seg:
                kept_seg_games += 1
                kept_seg_pos += len(seg)
            if is_clean_short_win(wseq):
                kept_mini_games += 1
                kept_mini_pos += len(wseq)
            u = seg | (set(range(len(wseq))) if is_clean_short_win(wseq) else set())
            union_pos += len(u)
            for i in seg:
                b = band(wp_of(wseq[i]))
                band_counts[b] = band_counts.get(b, 0) + 1

    dec = games or 1
    print(f"games read: {read:,} (malformed lines skipped: {bad}, "
          f"draws/non-decisive {draws:,}, decisive {dec:,})")
    print(f"segment-games: {kept_seg_games:,} ({100*kept_seg_games/dec:.1f}% of decisive)"
          f"  positions: {kept_seg_pos:,} ({100*kept_seg_pos/(plies_total or 1):.1f}% of plies)")
    print(f"clean-short-wins: {kept_mini_games:,} ({100*kept_mini_games/dec:.2f}% of decisive)"
          f"  positions: {kept_mini_pos:,}")
    print(f"UNION positions kept: {union_pos:,} "
          f"({100*union_pos/(plies_total or 1):.1f}% of decisive-game plies, "
          f"{100*union_pos/(read and sum(1 for _ in ()) or read) if False else ''}"
          f"{100*union_pos/(plies_total*read/max(read,1) or 1):.1f}% basis plies)")
    print("segment band mix:", {k: f"{v:,}" for k, v in sorted(band_counts.items())})
    per_game = union_pos / max(dec, 1)
    dec_frac = dec / max(read, 1)
    print(f"per-decisive-game retained positions (mean): {per_game:.1f}")
    est = per_game * 6_175_677 * dec_frac
    print(f"extrapolated to corpus (6,175,677 games, {100*dec_frac:.1f}% decisive): "
          f"~{est/1e6:.1f}M positions needing d20")


if __name__ == "__main__":
    main()
