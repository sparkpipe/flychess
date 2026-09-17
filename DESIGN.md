# FLYCHESS DESIGN — the architecture, the laws, the roadmap

*Canonical design doc. The monitor/driver automation reads this for context.
Updated 2026-09-17. Repo: sparkpipe/flychess (PR #1 = adaptive-stack-2026-09).*

## End goal

A chess-playing fly: a frozen Drosophila connectome (MaleCNS, 188,778 neurons,
~20M synapses) that **sees a board (raw planes), imagines forward (learned
world model), evaluates by comparing its chances against its opponent's
chances (the universal differential), and selects moves** — trained by a
gated curriculum with exact rule-derived labels, benchmarked by Stockfish
top-move agreement, with a grafted human memory/search module (H01) for the
opening book and internalized lookahead.

## Two-substrate architecture

| substrate | role | why |
|---|---|---|
| **Fly (MaleCNS, 188K N, ~20M unsigned synapses, frozen)** | vision → perception (reach maps) → legality/move intention → final reflex | fast 6-hop substrate; synapses frozen = biology as constraint |
| **H01 human cortex (57K N, 100M+ SIGNED DIRECTED synapses, laminated, recurrent)** | opening-book memory + internalized lookahead/search | cortical recurrence = deep sequential computation; E/I signs and direction come free |

The seam: a shared position encoding (raw planes + reach maps). The fly
perceives; the query projects into H01; H01 returns memory/lookahead; the fly
selects. Search is trained into H01 by amortized-search methods (deep-value
distillation, reply prediction, recurrence-as-thinking).

## The fly pipeline (the loop)

1. **Vision**: raw 13 planes (12 piece×color bitmaps + query marker) →
   sensory epithelium (trainable W_sens) → frozen connectome (settling-depth
   propagation) → perceived reach map. V1 graduated: 6/6 lessons, depth-2
   imagination verified 0.9977/0.9974.
2. **Imagination**: (planes_t, move) → planes_{t+1} + reach_{t+1} — the world
   model. V2 graduated: all 3 lessons; free-running chains verified 0.9977 /
   0.9974 in-distribution; dense-board extrapolation 0.963 (stage-5 boards fix).
3. **Selection**: for each legal move — imagine it (batched) → evaluate the
   imagined position with the **universal differential** minus the **null-move
   danger veto** → select argmax. The eval is fly-computed from its own
   imagined maps; the rig applies moves and never decides.

## The universal differential (the evaluation)

- **4096-slot option-space vector**: our move-intention profile minus theirs
  (theirs = one flipped-turn pass on the imagined position — the null-move
  construct, used ONLY where valid, see below).
- **Learned weights w per slot-dimension** = interpretable chess knowledge
  (king-safety dims, material dims, discovered-attack dims...).
- **Training ladder**: D1 rule-derived (material swing after their best
  recapture, procedural, exact) → D2 king-safety/structural (stage-5 mate
  semantics, rule-labeled) → D3 CC0 calibration (4.4M Lichess evals, LAST) →
  self-play refinement.
- **The zero-constraint**: mirrored/symmetric positions must score exactly 0
  (the equal-position detector, free labels by construction).

**The asymmetric objective** (operator scoring): win 1.0, draw 0.4 (White) /
0.45 (Black), loss 0. The fly maximizes winning percentage and takes
"illogical" complicating moves whenever the win/draw trade is favorable.

## The opening module (the operator's method, generalized)

The opening is a race: tempo → initiative → compounding tactics. All openings
are played with the same need for speed — the named gambits (Blackmar-Diemer,
Von Popiel, Smith-Morra as White; Budapest, Englund, Latvian as Black) are
the concentrated TRAINING EXEMPLARS, not a repertoire lock-in.

- **The initiative ledger**: forcing moves gained vs spent; dual-purpose moves
  (defends+develops, develops+attacks) score multiplicative.
- **The bands**: +1 gained move = edge; +2 = real compensation (precarious);
  +3 = compounding to a win; dropping to +1 = fighting for a draw.
- **The gambit frame**: −1 material is correct exactly when the ledger reads
  ≥+2 gained moves with the third reachable.
- **Corpora**: Morphy games (the canonical style source), the staged master
  parquet filtered per opening, the Lichess CC0 corpus for calibration.
- The operator's own result (near-IM performance with pure initiative style
  as an amateur) validates the style prior.

## The zugzwang law

The null-move construct is INVALID where pass is a real move — mutual
zugzwang exists at 13+ pieces. Therefore:
- The universal differential is **turn-independent**: both sides' coverage and
  threat maps compared structurally, no pass fiction. Tempo is a separate
  small correction.
- Null-move is demoted to **blunder prevention only** (the tactical veto).
- Endgames: tablebase-exact evaluation (212K pools staged, extending toward
  6-7 pieces) — the differential learns the zugzwang regions from tablebase
  labels, never from assumptions.

## The adaptive numerical stack (the laws)

1. **The saturation law**: at LEAK=0.5 the propagation rails 94% of neurons
   by step 3 — "6 levels" is functionally ~3 binary levels. The at-rail
   probe (fraction |a| ≥ CAP−ε per step) is THE health check: run it before
   blaming capacity, data, or architecture for any plateau.
2. **Per-step state normalization** (ANORM=1): rescale the state to the live
   band each step — the fix, validated (imagination: 0.954 railed-ceiling →
   0.98+; vision arms faster). Adopted for the v3 retrain.
3. **LayerNorm-affine / homeostatic variants**: Phase 1 arms racing the fixed
   normalization — verdict = which carries the occupied-cell semantics best.
4. **Per-neuron leak** (multi-timescale): Phase 2 graduated; lam
   differentiation begins (0.241–0.279, widening) — full-length rerun queued.
5. **Settling-based adaptive depth** (MPROP/VPROP/IPROP + MEPS/VEPS/IEPS):
   propagate until Δa < ε — simple boards settle fast, hard boards get depth.
6. **LP-weighted replay**: replay draws weighted by distance-to-gate
   (LAST_ACC registries) — the batteries nearest mastery rehearse most.
7. **Occupied-bin 25× loss weighting**: the ±1 cells (blockers, threats) are
   the chess payload; unweighted, the loss sacrifices them (measured 0-for-29).
8. **The trivial-pass law**: an init prior that enters the SCORE makes every
   gate vacuous (the w_pseudo2 lesson — fresh-model spot tests catch it).
9. **The spot-test asymmetry**: snapshot + new-regime crossing the ceiling =
   strong go; not crossing = inconclusive (the weights are regime-locked),
   so proceed to from-scratch regardless.

## The curriculum (stages, all gated at 0.98, scaffold-free, exact labels)

- **S1 movement** (6 lessons: rook/bishop/queen/knight/pawn+ep+promo/
  king+castle) — the vocabulary.
- **S2 blocking/capture** — rays stop at blockers; enemies are targets.
- **S3 protection** — grab undefended targets; refuse defended ones.
- **S4 king safety/pins/forks/discovered** — projected force, alignment
  concepts. Historical hard wall; the adaptive stack is the attempt.
- **S5 mates** — mate-in-1..3 = imagination-for-depth + check semantics;
  100% gate.
- **S6 Dvoretsky endgames** — tablebase-gated state-change training;
  zugzwang-correct (no null-move); 212K pools staged.
- **The differential lessons** (D1 material, D2 king safety, D3 CC0) —
  the eval layer, per the training ladder above.
- **The opening module** — the race/initiative/gambit-compensation families
  per the operator's method; Morphy corpus + master parquet; exemplar
  systems = the named gambits; style prior = maximize winning chances.

## The verification stack

- **The loop rig** (fly_loop_rig.py): closed self-drive; imagination accuracy
  per depth (in-distribution + dense extrapolation); no-hang rate with/without
  imagination (the ablation); counterfactual pairs; mate-in-2 stretch probe;
  Stockfish top-move agreement (the external ladder).
- **The at-rail probe**: before any training campaign, on any substrate
  (including H01).
- **Spot tests are asymmetric**: snapshot + new-regime crossing the ceiling =
  strong go; not crossing = inconclusive → from-scratch proceeds anyway.

## The H01 graft (in flight)

- Pulled: /mnt/model-warm/human-h01-connectome (910G, verified, PUBLISHED;
  cold-archives nightly). Source: H01 (Shapson-Coe et al., Science 2024),
  1mm³ human temporal cortex (BA21), ~57K neurons, 100M+ signed directed
  synapses, laminated.
- **Recon done**: c3/synapses = neuroglancer annotations v1 (LINE), sharded
  (murmurhash3, preshift 10, minishard 13, shard 5, gzip), uint32 type enum
  {1: inhibitory, 2: excitatory} + pre/post_synaptic_cell relationship shards;
  spatial0/0_0_0 = the coarse chunk (362KB, compressed custom binary);
  python 3.14 on the node has no cloud-volume wheels → py3.12 uv venv at
  ~/chess-lab/.venv312 with cloud-volume 12.14.4 installed; info manifests
  were served gzipped — gunzipped in place.
- **Experiments** (operator order): (1) book-moves memory — train H01 on the
  opening book (position → book move, exact labels); (2) Leela-type search —
  the amortized-search stack on cortical recurrence.
- The saturation protocol runs on H01 first (its own at-rail profile).

## Current state (2026-09-17)

- v3+ANORM retrain: L1 rook (113) / L2 bishop (136) / L3 queen (step 1) /
  L4 knight (in progress, 0.92+ climbing) — the full-stack A/B in flight.
- V1 vision: COMPLETE (6/6, nz 1.0 late pieces). V2 imagination: GRADUATED
  (both normalized + control arms; free-running chains verified).
- Phase 1 arms (LN-affine, homeostatic): in I3 chains, nz 0.975-0.985.
- Phase 2 (per-neuron leak): graduated short-protocol; full rerun queued.
- H01: published on warm, cold-archive tonight; recon complete; extraction
  script next (h01_extract_full.py: bbox-grid → checkpointed edge parts;
  cloud-volume in ~/chess-lab/.venv312; the info manifests gunzipped; the
  np.hstack cloud-volume bug shimmed; Bbox uses .minpt/.maxpt).
- PR #1 open (adaptive-stack-2026-09 → master): merge when the imagination
  arms + stages 2-4 complete under the new regime.

## The laws (hard rules)

1. Scaffold-free gates only. 2. Spot tests are asymmetric evidence. 3. The
at-rail probe precedes every campaign. 4. Stable seeds everywhere. 5. Tiered
acceptance: 0.98 pass / 0.975 marginal / <0.95 fail. 6. The v3 chain is never
killed mid-lesson; maintenance runs inside the loop. 7. pkill patterns run
ALONE in their own ssh. 8. Patches: local canonical file → assert-guarded →
smoke → ship. 9. PR flow: branch → test → PR → merge at stability.
10. Verify training liveness with fresh tails, never stale reports.

## Ops addendum — H01 extraction relaunch (2026-09-17, fix-as-found)

The driver's H01 relaunch line must run the VENV python:
`cd ~/chess-lab && H01_TARGET=12000000 nohup ~/chess-lab/.venv312/bin/python3 h01_extract_full.py > h01_extract.log 2>&1 < /dev/null &`
— bare `python3` is system 3.14 (no cloud-volume wheel; instant ModuleNotFoundError).
Probe liveness with `pgrep -f h01_extract_full` (a `h01_extract[.]py` pattern
never matches `h01_extract_full.py` — false DOWNs). If a relaunch dies with
`OSError Errno 28` (disk full on /), clear `~/.cache/pip ~/.cache/uv` first
(kept ~1.4G on 2026-09-17; leave ~/.cache/huggingface — the champion lane's
fetchers need it). The enumeration phase re-runs from shard 00 each relaunch
(~1 min/shard, by_id shards ~72MB); edge parts in h01_edges/ are checkpointed
and idempotent-skipped.
