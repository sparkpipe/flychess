#!/usr/bin/env python3
"""Per-source decomposition of the piece_down bin label problem.
The fleet val bin's rows are in export order: OTB seg rows first, then
puzzle rows, then miniature rows (counts from the fleet manifest). Compare
each slice against the clean SF17-d12 relabels (row-aligned, verified):
which source carries the corruption?"""
import struct

ORIG = "/srv/workspace/chess-active/bins-fleet-v1/piece_down.val.bin"
MINE = "/srv/workspace/chess-active/pd-depthexp/bins/d12.val.bin"
# manifest: otb_seg_val 46121, puzzle_val 82708, miniature_val 7766
SLICES = [("otb_seg", 0, 46121),
          ("puzzle", 46121, 46121 + 82708),
          ("miniature", 46121 + 82708, 136595)]


def read_bin(path):
    raw = open(path, "rb").read()
    n = len(raw) // 40
    out = []
    for i in range(n):
        sfen, score, mv, ply, b, u = struct.unpack_from("<32shHHbB", raw, i * 40)
        out.append(score)
    return out


orig = read_bin(ORIG)
mine = read_bin(MINE)
print(f"records: orig={len(orig):,} mine={len(mine):,}")

for name, lo, hi in SLICES:
    o, m = orig[lo:hi], mine[lo:hi]
    n = len(o)
    flips = sum(1 for a, b in zip(o, m) if a * b < 0)
    deltas = sorted(abs(a - b) for a, b in zip(o, m))
    big = sum(1 for d in deltas if d > 100)
    near = sum(1 for d in deltas if d <= 20)
    mean_d = sum(deltas) / n
    # magnitude ratio: are orig labels compressed/expanded vs truth?
    mag_o = sum(abs(a) for a in o) / n
    mag_m = sum(abs(b) for b in m) / n
    print(f"\n[{name}] rows {lo:,}..{hi:,} (n={n:,})")
    print(f"  sign flips: {flips:,} ({100*flips/n:.1f}%)")
    print(f"  |delta|: median {deltas[n//2]}  p90 {deltas[int(n*0.9)]}"
          f"  mean {mean_d:.0f}  >100cp {100*big/n:.1f}%  <=20cp {100*near/n:.1f}%")
    print(f"  mean |label|: orig {mag_o:.0f} vs clean {mag_m:.0f}"
          f"  (scale ratio {mag_o/mag_m:.2f})")
