# Mapping-plateau experiment matrix

Theory (operator): the feature->neuron mapping is the limiting factor; the
fly plateaus where the mapping can no longer support the stage's demands.

## Protocol
1. Baseline: current mapping (full-sensory injection 26,933 -> 2-step prop
   -> RANDOM 4096-slot readout). Run stages until gates stop passing.
2. Plateau = stage where gate stalls (record curve: fly_cb_log.jsonl).
3. Variants (one parameter at a time), each in TWO arms:
   - incremental: warm-start from the plateau checkpoint
   - from-scratch: FRESH=1
   Gate result decides. Re-run the full stage battery.

## Variants
V1 readout=variance        (activation-variance-selected 4096)   [ready]
V2 propagation depth       2 -> 3, 4
V3 leak                    0.5 -> 0.3 / 0.7
V4 injection site          full-sensory vs CX vs MB vs combined
V5 modality wiring         feature groups -> specific glia/channels
V6 gain schedule           injection amplitude scaling
V7 dual-lane blending      mid/end evaluator mixture weights

## Baseline findings
- Stage 1 (piece-movement legality): top1=1.00, PAIRWISE stuck ~0.71
  (gate 0.99). Loss plateaued at ~0.038. Mapped limiter candidate: random
  readout under-receives legality structure (same failure class as the CX
  dilution bug) -> V1 is the first test.

## Wiring measurements (wiring_engineer.py, wiring_report.json)
- Capacity = 1.0 at ALL sites (16 channels multiplexed perfectly through 2
  linear steps) — channel count is NOT the limiter.
- Routing: sensory_periph is the WORST hub (gain 105K, 50% self-locked).
  medulla_Tm/lobula/lobula_plate = high-gain visual path; LH routes INTO KC.
- 21 dead features (king-count constants, empty-slot pins) — flagged pre-train.
- V2 = structured wiring: 9 families -> measured sites, direct site injection
  (bypasses periphery), block-masked weights, variance readout.
  arms: fly_cb_v2wire.pt (fresh) vs fly_cb_v1var.pt vs baseline plateau.

## New-signal principle (operator)
Adding signal means NEW information, not repeating the same channel. The
color-blind relative view collapses positions except at the boundary — so
the EDGE OF BOARD is genuinely new signal. Added: per-piece distances to all
4 edges (eye{i}_eu/ed/el/er, 128 dims) + attacked-square-on-edge conjunctions
(atk_edge_my/their, 128 dims) → 2746 total. Mirror-invariance verified 0-diff.
The from-scratch verification run trains with these; the rook rank-1
failures were exactly this missing concept.

## Image-view contract (parked, operator ruling 2026-09-16)
Image planes OFF until combined tactics stage. When re-added:
- normalize image planes to unit variance vs the abstract channel scales
  AT INJECTION TIME (neither view may dominate the other)
- re-test at stage 4+ (forks/discovered = where raw detail may add capacity)
- stage-1 verdict: image view slowed rook 87->218 steps, zero gate benefit
