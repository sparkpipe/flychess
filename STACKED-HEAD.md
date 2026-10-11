# STACKED HEAD — design proposal (NOT yet built; awaiting operator clearance)
**2026-09-29. The operator's technique: frozen experts' raw outputs + normal features -> a head trained by self-play.**

## 1. Data flow (inference)

```
position
  -> ALL 14 experts' feature transformers each emit their 1024-dim
     activation vector (operator design: every expert used — no top-k)
  -> per-expert calibration (fixed affine, computed once)
  -> HEAD: concat(14 x 1024 calibrated, hand features) -> eval (cp)
```

- **Which layer:** the feature-transformer output (the input to L1) — the
  richest internal representation; the scalar head of each expert is dropped.
- **ALL experts, always (ruling 2026-09-29):** 14 x 1024 = 14,336 head inputs.
  The head learns per-context trust (irrelevant experts -> ~zero weight).
  Inference cost: 14 transformer passes per eval (~10x eval cost inside
  search — acceptable during RL self-play; removed by distillation for the
  final artifact).

## 2. Calibration (one-time)
On a 100K common-position sample: per expert, per-dimension mean/std of the
1024 activations; head sees z-scored inputs. Removes the scale mismatch
(tactics net vs balanced net activations differ wildly). Stored as constants.

## 3. Head architecture (proposal)
- Input: 14x1024 calibrated + HAND FEATURES, two tiers (operator chooses):
  - Tier 1 (~90 dims, all already computed in our pipeline): per-side piece
    counts (10), material config one-hot (13 residue classes), men, ply, stm,
    castling rights (4), ep flag, lock_c/lock_e/tension/open (4), bishop
    pair (2), opp-bishops flag, per-file pawn counts (16).
  - Tier 2 (~300-1500 dims, derivable offline + in-engine C++): per-piece
    threat counts/attack tables, mobility per piece, pawn islands/doubled/
    isolated/passers per file, king attack zones, (the fly-era threat maps
    are deprecated but the computations are standard).
- **Linear head first** (one dot product, ~4K params — cheap, hard to
  overfit, directly interpretable as learned expert trust); **MLP-64 variant
  second** if linear underfits validation.
- Output: cp (trained against the same win-prob sigmoid loss the experts used).

## 4. Training, two phases (progress visible; stop when not improving)
- **Phase A — supervised bootstrap (no engine needed):** train the head on
  the audited bins (5.6M positions, depth-12 cp targets already in the
  records). Held-out 5% validation; train while val loss improves, stop on
  3-check plateau (the operator's "continue as long as it makes a better
  model, don't overtrain"). Minutes on the 5090 at ≤10GB.
- **Phase B — self-play RL (the operator's recipe):** engine plays games with
  the head; positions+outcomes train the head (the 1-hour/2400 pipeline
  pattern, head-only learning). Iterate: gen (sparks, ≤10 CPU/node, or 5090
  CPU) -> train -> reload -> repeat. Continue while match performance
  improves (SPRT gates between iterations), stop on plateau.

## 5. Engine integration
Fork gains: load N expert nets + head weights file; evaluate() = top-k
transformers -> calibrated concat -> head. The 13-slot machinery stays.
Head quantized (int16/int8) for speed. (~200 lines C++; done AFTER the
background agent lands its crash fix — no concurrent fork edits.)

## 6. Evaluation protocol
Ladder vs SF at fixed skills (fastchess), then SPRT vs (a) hard-routing MoE,
(b) monolithic control. If the head wins: distill the whole stack into one
net (the endgame that recovers 1x eval speed).

## 7. What needs your ruling
1. Hand-feature tier: Tier 1 (~90) vs Tier 1+2 (~300-1500).
2. Linear-first head vs MLP-first.
3. Phase B self-play pool: sparks at ≤10 CPU vs 5090 CPU first.
4. Distillation as automatic follow-up or separate decision.
