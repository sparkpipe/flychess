#!/usr/bin/env python3
"""Dry-run of the nQ segment rules (extractor v2) over the d20 trajectory
corpus — counts exactly what the d20 eval-slope segments will contribute.
Rules verbatim: winner-pov cp -> wp = sigmoid(cp/361); bands collapse
0-30 / defend 30-45 / equalize 45-55 / press 55-70 / convert 70-95 / win
95+; blunder boundary |dcp| >= 120 (cp space); same-band (draw-to-draw)
segments EXCLUDED; games with missing evals dropped whole."""
import math
import sqlite3
from collections import defaultdict

con = sqlite3.connect("file:/srv/workspace/chess-active/store/games.db?mode=ro",
                      uri=True)

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


games = {}
for gid, off, result in con.execute("select gid, off, result from games"):
    games[gid] = (off, result)

rows = con.execute(
    "select p.ref, p.fen, l.cp from positions p"
    " join labels l on l.fen = p.fen and l.engine='sf17' and l.depth=20"
    " order by p.seq").fetchall()
print(f"positions with d20: {len(rows):,}")

by_game = defaultdict(list)
for ref, fen, cp in rows:
    gid, ply = ref.split(":")
    by_game[int(gid)].append((int(ply), fen, cp))

kept = band_counts = defaultdict(int)
kept = defaultdict(int)
seg_lens = []
dropped_games_missing = 0
excluded_same_band = 0
total = included = 0

for gid, plies in by_game.items():
    off, result = games.get(gid, (0, "1/2-1/2"))
    if result not in ("1-0", "0-1"):
        continue                      # draw games carry no winner-pov
    winner_is_white = result == "1-0"
    plies.sort()
    # winner-pov cp for the side to move -> winner perspective
    seq = []
    ok = True
    for ply, fen, cp in plies:
        stm_white = fen.split()[1] == "w"
        same = (stm_white == winner_is_white)
        seq.append((ply, cp if same else -cp))
    # blunder boundaries + band segmentation
    segs = []
    cur = [seq[0]]
    for prev, curp in zip(seq, seq[1:]):
        if abs(curp[1] - prev[1]) >= 120:
            segs.append(cur)
            cur = []
        cur.append(curp)
    segs.append(cur)
    for seg in segs:
        wps = [wp_of(c) for _, c in seg]
        b0, b1 = band(wps[0]), band(wps[-1])
        if b0 == b1:
            excluded_same_band += len(seg)
            continue
        for (ply, cp), wp in zip(seg, wps):
            kept[band(wp)] += 1
            included += 1
        seg_lens.append(len(seg))
    total += len(seq)

print(f"winner-decided games: {len(by_game):,}")
print(f"segment positions INCLUDED (band start != end): {included:,}")
print(f"excluded same-band (draw-to-draw): {excluded_same_band:,}")
print("per-band (current band of each included position):")
for b in ("collapse", "defend", "equalize", "press", "convert", "win"):
    print(f"  {b:9s} {kept[b]:,}")
if seg_lens:
    seg_lens.sort()
    print(f"segments: {len(seg_lens):,}  median len {seg_lens[len(seg_lens)//2]}"
          f"  mean {sum(seg_lens)/len(seg_lens):.1f}")
