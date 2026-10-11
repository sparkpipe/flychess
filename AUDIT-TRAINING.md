# MoE TRAINING AUDIT — data + protocol review before training (2026-09-28)
*Companion to DESIGN-MOE. Every claim traceable to the 100% bin audit or the census.*

## 1. Data inventory (per-position classification, ruling (b); v2 repack in flight)

| slot | bin (+aug) | records (v1 floor; v2 ≥ these) | verdict |
|---|---|---|---|
| tactics | tactics.bin | 23,336,332 | strong |
| balanced L0 | +aug | 3,084,551 | strong |
| N vs B | +aug | 1,174,088 | strong |
| balanced L1 | +aug | 944,222 | strong |
| exchanges | (dual) | 522,447 | strong |
| balanced L2 | +aug | 195,348 | workable |
| opp-bishops | +aug | 129,250 | workable |
| B vs R | +aug | 56,753 | thin — over-sample epochs |
| dvoretsky | | 49,832 | thin — grows in v2 (boundary crediting) |
| N vs R | +aug | 45,451 | thin |
| balanced L3 | +aug | 37,222 | thin |
| R vs 2 minors | +aug | 31,580 | thin |
| Q vs material | +aug | 15,772 | very thin — candidate for fallback-to-nvb at loadout |
| tablebase | syzygy | 5,000,000 | done |

Reference points: 14.6M curated positions → ~2000-2200 Elo; 102M self-play → ~2400.
Thin bins (<100K) cannot train a strong standalone net in one pass — mitigations
below (§4). No shortcuts taken: every number above is the audited 100% count.

## 2. Data integrity (100% audit, corrected auditor)
- Class verification: per-position repack (v2) targets **0 mismatches** — v1 showed
  only piece-attribution drift (now ruled out) and audit-side key bugs (fixed).
- tactics/aug bins: 0 mismatches already; dups = opening transpositions + deliberate
  aug weighting (quantified per bin in audit_report.txt).
- Score convention: stm-perspective cp — identical to the pipeline that produced the
  2400-Elo baseline; consistent across every bin (verified by unpack audit).
- Effects coverage after v2: boundary crediting admits the trade-completion positions
  (captures/en-passant/promotions at material changes) into the main bins — the v1
  capture-poverty (239 captures in 2.67M L0) is corrected by design, not sampling.
- Eval noise: depth-12 wall fuzz ~6% ACCEPTED (ruling); cp targets dominate.

## 3. Engine integration (fork v1.12, committed ede833f)
- 13 slots; router = the SAME predicates as the packer's assign() (residue classes,
  opp-bishops, dvoretsky, exchanges, balanced shards); route-verified on crafted FENs.
- Bins and router now agree per-position (this was the point of ruling (b)).
- Fallback: unloaded/weak slots fall back to the first loaded net — the loadout plan
  uses this for very-thin slots if their nets underperform.
- Dual-signal tactics protocol (tactics-first, tree-until-quiet, positional-verifies-
  leaf): NOT yet in the engine — separate work item after the first MoE SPRT; the
  tactics net still trains now and loads into its slot.

## 4. Training protocol proposal (needs operator sign-off)
- Trainer: nnue-pytorch, features HalfKAv2_hm^, 40-byte bins, CUDA on the 5090
  (20GB VRAM approved). One run per expert over <expert>.bin + aug_<expert>.bin.
- **Equal-total-budget law**: MoE arm total sample-passes = monolithic control's.
  Control: one net over the concatenated qualifying data (same selection rules).
  Concrete split: proportional-to-bin-size epochs, each expert capped so that
  Σ(bin_i × epochs_i) = B = control's total passes. Thin bins get a minimum-epochs
  floor (proposal: enough passes for convergence, ~the L0 per-position pass count
  at 1/8 scale) — the floor consumes budget that would otherwise over-train big bins;
  the total stays B.
- Serialization → EvalFile1..13; loadout; PHASE_MOE=1 routing.
- Match: SPRT vs the monolithic control at B; then 2B/4B curves (ruled program).

## 5. Known risks / open items
1. qvmat (15.7K): likely below trainable floor — plan: train, then decide by quick
   match whether to keep or route those positions to nvb (operator call).
2. Class noise at walls (~6%): accepted; affects expert boundaries only.
3. Both-sides duplicates of winner data: the both dump includes winner positions
   already in the winner-only audited set — v2 bins are built from the BOTH stream
   only (no double count). ✓
4. Tactics engine-side gate: post-SPRT work.
5. The 21,870 unresolved-color gambit games: available via a PGN pass if ruled.
