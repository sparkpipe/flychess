# DATA CONVENTIONS — flychess training data (the law + the incident)

## The law

**Bin scores are ALWAYS side-to-move (stm) perspective.** The nnue-pytorch trainer's
loss compares the bin score directly against the net's stm-pov output
(`calculate_sf_loss` in `model/nnue.py`; upstream Stockfish training-data convention).
The nQ-era packer documented this: `pack_expert_bins.py` line 12 —
"Score: stm-perspective cp as stored by the eval worker."

**Every new label source MUST pass `source_audit.py` (sign-match vs SF17, split by
stm/white) before it may be packed.** No exceptions — this gate exists because its
absence cost the entire nQ16 generation (below).

## Measured conventions of every source (2026-10-07 audit, SF17-d12 ground truth)

| source | path | volume | measured POV | pack with |
|---|---|---|---|---|
| OTB segments | `wp_fit/segments/*.tsv` (cols: fen\|cp\|stm\|class\|…) | corpus | **stm** (31/36 sign) | verbatim |
| lichess puzzle evals | `~/backlog_shards/puzzle_eval/w*.tsv` (cp12 col) | 8.69M | **stm** (65/74) | verbatim |
| miniature labels | `chess-active/miniature_labels.tsv` | 43.5M | **WHITE** (39/40) | `pov="white"` |
| DEGM book | `wp_fit/trainsets/sources/degm.tsv` (cp×100) | ~6k | white (24/40, coarse) | `pov="white"` |
| syzygy WDL | `wp_fit/trainsets/sources/tb.tsv` | ~100k | winner-sign (not an eval) | wp-inversion cp, per design |

Label engines: segments/puzzles = original eval-worker (stm). Miniatures = **SF8**
(`/usr/games/stockfish`, mini_label.py line 4) flipped to white-pov at line 41-42.

## The nQ16 incident (2026-10-02 → 2026-10-07)

`assemble16r.py` / `assemble16anti2.py` packed all sources verbatim. The miniature
labels (43.5M rows, the largest source, routed overwhelmingly into the opening and
middlegame experts) are white-pov, so **every black-to-move miniature row taught the
inverted eval** — ~half of the dominant training signal.

Symptoms (all explained, all measured 2026-10-07):
- n16 evaluates a queen-up position at +246cp (nQ13: +1069) — 4–7× material
  compression; worst on black-to-move (+142 error vs −900 truth).
- Play: drops pieces and exchanges ("playing backwards") — material looks nearly free.
- val_corr stayed positive (0.5+ on opening experts) because the val split shares the
  flip — self-consistent labels, miscalibrated truth (oracle-mirrors-module trap).
- Bin-sample audit: piece_down labels at 0.34× true scale, qvmat/tb sign-chaos.
- Result: nQ 6–0 n16 at nps-calibrated time (1s vs 1.14s).

Fix: `emit(..., pov=...)` perspective normalization in `assemble16r.py` +
`assemble16anti.py` (white-pov sources negated for black-to-move at pack time —
no relabeling needed; labels are consistently white-pov). Bins must be re-assembled
and experts retrained from the fixed bins.

## Recreating nQ (13-expert) training from scratch — YES, fully

Every input survives on cold RAID + repo:
1. Sources: segments (`wp_fit/segments/`), puzzles (`~/backlog_shards/`), DEGM, syzygy.
2. Packer: `chess-lab/tools/pack_expert_bins.py` (13-expert cascade, stm-pov, the
   proven nQ packer).
3. Trainer: `nnue-pytorch/` (box + repo); feature set `Full_Threats+PP_3Wide+HalfKAv2_hm^`.
4. Engine + router: `chess-lab/tools/nq16/nq13-engine/` (13-slot, EvalFile..EvalFile13;
   wiring per `match16/match_nq_n16.sh`: EvalFile5=nvb [slot "nvsb"], EvalFile13=tactics
   [slot "tb"]).
5. Train per expert on its bin (skip=3, no limit, hourly ckpt, watcher exports .nnue).
Not bit-reproducible (epoch history/seed of the original runs), recipe-complete.

## Recreating nQ16 correctly

Same as above with: router `routerB.py`, assembler `assemble16r.py` (FIXED, pov-aware),
16-expert taxonomy `EXPERTS16.md`, engine `n16-engine/`. After re-assembly, re-run
`match16/audit_bin_labels.py` on the new bins (expect: stm-sign ≈ 100%, scale ≈ 1.0
on full-cp sources) BEFORE launching any training.
