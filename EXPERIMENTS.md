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
