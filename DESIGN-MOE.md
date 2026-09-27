# PHASE-MoE DESIGN DOCUMENT — consolidated from the full operator record
**Version 1.0 — 2026-09-27. This document is the single source of truth for the expert taxonomy, routing, data selection, and experiment protocol. Every line traces to an operator ruling. Where a reading was chosen, it is marked [READING — ruling pending].**

---

## 1. Goal

World-champion chess bot via Stockfish search + our NNUE evaluation, partitioned into
specialist nets routed by material configuration and phase. **Path A**: fork Stockfish,
replace the NNUE with our nets trained on curated OTB data (never self-play), route by
piece configuration.

Standing directives (violations are program-stopping):
1. No unauthorized experiments.
2. No shortcuts: no depth-1 evals, no partial dumps, no monolithic tests dressed as the experiment.
3. No lying or sugar-coating; numbers exact (380K/10.3M = 3.7%, not 20%).
4. Curated OTB data only for the real training.
5. The critical comparison: **MoE vs monolithic at the SAME TOTAL training budget** (plus 2x/4x/8x budget curves).
6. Depth 12 minimum for all evaluation work.

---

## 2. Architecture

- Forked Stockfish (`phase-moe-v0` branch on rtx5090, `/home/spec/Stockfish`), array-based
  N-slot engine: per-slot `EvalFile`, per-slot replicated `Network`, per-slot
  `AccumulatorStack` + `AccumulatorCaches` (one dirty-set mirror per move; only the routed
  net evaluates — zero routing overhead measured in v0).
- UCI: `EvalFile` (slot 1) + `EvalFile2..EvalFileN` for the rest; unfilled slots fall back
  to the first loaded net (never a zero-weights eval).
- Router = pure function of the position (material counts, pawn contact, total men, game ply).
  Cut points are UCI options calibrated from data, never frozen in code.
- Debug modes (`PHASE_MOE` env): 0=off, 1=route, 2=always-endgame, 3=dual-evaluate, 4=route trace.
- Trainer: official nnue-pytorch, features `HalfKAv2_hm^` (with caret), 40-byte nodchip
  bins (`<32shHHbB`, unsigned move field).
- Slot count is one constant; the plumbing carries any N.

---

## 3. THE EXPERT LIST

### 3.1 Phase / structural experts

| # | Expert | Description | Routing predicate | Training data |
|---|--------|-------------|-------------------|---------------|
| 1 | **opening-gambit** | Early sacrificed or grabbed material; initiative-first play. Dataset ALREADY EXISTS (gambit pool mined from 94K matched games; initial-drop-of-a-pawn pattern). | game_ply < opening bound AND asymmetric material | existing gambit dataset + early-asymmetric segments |
| 2 | **balanced** | Symmetric material, any pawn count, queens on or off, no listed confrontation. The locked-pawn closedness spectrum lives INSIDE this domain (see §5). | symmetric counts, no confrontation fired | balanced-config segments, all trajectories |
| 3 | **piece-trades** | The trade decisions: positions with an equal-value capture available, and the exchange moves themselves (material-change boundaries). The expert in WHEN to trade. | equal-type or cross-minor capture available in non-endgame material [READING — ruling pending] | `exch`-tagged positions + pre-trade tension positions |
| 5 | **endgame-dvoretsky** | Practical endgames, 6-10 men, not a listed confrontation. CONFIRMED by operator. | 6 <= men <= 10 (after confrontation table) | dvoretsky-region segments |
| 6 | **endgame-tablebase** | Exact endgames, <=5 men. CONFIRMED. Syzygy-generated training data COMPLETE (exactly 5,000,000 positions in `tb_training.bin`). | men <= 5 — absolute, first check | syzygy data (done) |

### 3.2 Tactics expert (independent, puzzle-trained)

| # | Expert | Description |
|---|--------|-------------|
| 7 | **pure tactics** | ONE independent expert trained on the tactics puzzles (Lichess DB, 6.1M theme-tagged) — regardless of material balance, eval trajectory, phase. It only sees pure tactics. It exists OUTSIDE the MoE: see the dual-signal architecture (§2.1). Data: `/home/spec/chess-lab/puzzles/lichess_db_puzzle.csv`. |

**Dual-signal player (operator ruling, refined):** the structured-strategy MoE plays
general-concept chess — eval slope improvement, slowly squeezing to checkmate. The tactics
player cares only about the immediate tactical win.

**Protocol (verbatim, assembled):**
1. **The tactics evaluator runs FIRST** — on every position, at evaluation time.
2. It **finds its best move** (the tactical candidate — the puzzle-trained net is
   move-producing, not just an eval).
3. Behind that move **there is a move tree at depth** — the tactical line. **The leaf
   node** is the ending position.
4. The **positional evaluator evaluates the leaf node** (depth TBD).
5. **If the positional evaluator says it is a big jump in win percentage — THAT is the
   trigger:** immediate tactical win confirmed, the tactics move is the answer. Done.
6. **No jump: no tactic** — normal positional MoE evaluation/routing.

**Tactical tree termination (ruling):** the tree expands **until the position is QUIET** —
no imminent trades, no other short-term tactics. Quiescence termination, not a fixed ply
count. (Same notion as the trade/tension data tags: the tree runs through the forcing
material and stops when the dust settles.)

**Leaf positional depth (ruling):** the depth of the positional search at the end of the
tactical leaf is a **RUNTIME PARAMETER** (UCI option).

**Time management (ruling) — trajectory-driven:**
- **Investment phase:** early, we invest our time to get as good of a position as possible.
- **Harvest phase:** at some point we switch to "make reasonable moves quickly" — the hope
  is the time investment that got us into time trouble created a good position.
- **Critical-area rule:** if the position is LOSING win percentage and getting close to the
  critical area **40%**, we spend MORE time to try to get back to **45%+**.
- Constants given: 40% critical, 45% recovery. Investment→harvest switch point: runtime
  parameter. The win% trajectory over the game (EMA'd) drives the policy — same trajectory
  signal as the training-data bands (45-55 equalize, 30-45 defend).

TBD calibration knobs: jump size/reference for the trigger.
The tactics expert is puzzle-trained (pure tactics, blind to material balance/eval
trajectory/phase) and runs first at every evaluation.

### 3.3 Material confrontation experts (operator's list, verbatim)

Traded-down technique confrontations. Matching reading [READING — ruling pending]: a class
fires when the board's non-pawn material, after canceling common pieces, IS the confrontation
(pawns free) — so confrontations inside big material belong to the tactics family, and the
confrontation experts own the reduced/technique positions.

| # | Expert | Confrontation |
|---|--------|---------------|
| 8 | **N vs B** | knight vs bishop minor ending (2 minors on board) |
| 9 | **2N vs N+B** | two knights vs mixed minors (3 minors on board) |
| 10 | **2B vs 2B** | bishop pair vs bishop pair |
| 11 | **N vs R** | knight vs rook — INCLUDES exchange-down play (both sides: the rook converting, the minor side saving the draw — "saving the draw being an exchange down is a super important skill") |
| 12 | **B vs R** | bishop vs rook — same, includes exchange-down |
| 13 | **2N vs R** | rook vs two knights |
| 14 | **N+B vs R** | rook vs knight+bishop |
| 15 | **2B vs R** | rook vs bishop pair |
| 16 | **2R vs Q** | two rooks vs queen |
| 17 | **R+N vs Q** | queen vs rook+knight |
| 18 | **R+B vs Q** | queen vs rook+bishop |
| 19 | **2R vs 2R** | heavy-piece symmetric — own technique class |
| 20 | **opposite bishops** | equal bishops, opposite square colors — drawing/technique patterns |

Exchange-down is NOT a separate expert — covered by N vs R / B vs R (operator ruling).

Starved classes (rare confrontations with too little OTB mass) fold into a neighboring class
ONLY on explicit operator ruling, with the mass numbers shown first (e.g. the three
rook-vs-two-minors forms). Never silently.

---

## 4. Routing cascade (first match wins)

1. men <= 5 → **tablebase** (exact knowledge is absolute)
2. confrontation table (§3.3) → its expert
3. men <= 10 → **dvoretsky** (general practical endgame not in the table)
4. tactics family: unbalanced material inside big material → matrix-derived bucket
5. queens == 0, symmetric → **mid-queenless**
6. equal-value capture available → **piece-trades**
7. game_ply < opening bound AND asymmetric → **opening-gambit**
8. symmetric → **balanced** (spectrum inside, §5)

The opening boundary ply is a UCI option, calibrated from data. No other free constants.

---

## 5. The locked-pawn closedness spectrum (balanced positions only)

Applies ONLY to balanced (symmetric-material) positions. Components, per the operator's
definition:

- **Locked**: facing pawns on the same file (white pawn directly below a black pawn).
- **Tension**: capturable pawns (a pawn of one side attacks a pawn of the other).
- **Open file**: no pawns at all on the file.
- **Edge files locked != closed**: center remains playable — edge locks carry half weight.
- **Before any contact**: neutral/undetermined (e.g., the start position).

Calculation (per position, fully position-derivable):
```
per file f in a..h:  weight w(f) = 2 for c,d,e,f; 1 for a,b,g,h
L = sum over locked files of w(f)            # directly facing pairs
T = number of pawn-attack pairs (tension instances)
O = sum over fully-open files of w(f)

if L + T + O == 0:  NEUTRAL (no pawn contact yet)
else:               C = (L - O) / (L + T + O)   in [-1, +1]
                    C -> +1 fully locked center, C -> -1 fully open
```
Extractor v2 already emits per-position lock_c (center locked files), lock_e (edge),
tension, open — so C is computable on every training position and engine-side at eval time.

**How a spectrum gets trained** (the open question, answered):
You do not train the axis itself. The NNUE input already encodes every pawn's square —
closedness is a function of the input, so ONE balanced net learns closed-vs-open evaluation
differences from pawn structure. Splitting the axis into separate nets (e.g. closed/semi/open
at calibrated quantiles) is available if the matrix shows the regions are evaluationally
disjoint enough to justify capacity concentration — but hard cuts on a continuum create eval
discontinuity at the boundaries (routing jitter), so the split needs evidence AND a ruling.
v1: one balanced net; C is reported in the matrix and available as a routing axis later.

---

## 6. Training data selection

- **Source**: `gambit/filtered.pgn` — decisive OTB games, winner >=2400, loser >=2000
  (~1.53M games). **BOTH SIDES' positions (ruling 2026-09-27: "the losing side can also
  create training worthy data" — defensive play, exchange-down saving, is the DOWN side's
  data).** Full archive (10.3M games) is the expansion source for starved classes.
- **Evaluations**: depth 12 minimum (fleet, all 14 sparks). Depth-1 gambit evals are OBSOLETE, never used.
- **Moves included**: castling, en passant, threefold repetition, underpromotions — all must appear.
- **Elo floor**: stay with 2400+ for our moves.
- **Segmentation** (extractor v2, built and validated):
  - Blunder boundary: eval swing >= 120cp in cp space (the v1 wp-space bug is fixed).
  - **Sub-division at every material change**; each sub-segment trains its own expert.
  - Exchange-move positions tagged `exch` (piece-trades data).
  - Per-position material config + opp/same bishops + lock_c/lock_e/tension/open + men + phase.
- **Trajectory classes** (win-probability bands, W = 1/(1+exp(-cp/361))):
  - press 55-70, convert 70-win, equalize 45-55, defend 30-45, win 95+, collapse <30.
  - Segments classified start-band -> end-band.
  - **EXCLUDE draw-to-draw / same-band segments** (no skill demonstrated).
  - **INCLUDE losing-game segments** (improving play is valuable).
  - Eval slope computed with an EMA (operator ruling).
- **The matrix**: (material config x trajectory) with segments/ply counts over the full dump.
  Roles: (a) tactics-family bucket derivation, (b) per-expert data mass, (c) calibration of
  every UCI cut point, (d) starved-class detection.

---

## 7. Pipeline (state as of this doc)

| Step | State |
|------|-------|
| Full OTB dump on filtered.pgn | RUNNING (~18M of ~63M positions, ~9.1K pos/s) |
| TB specialist data | COMPLETE — exactly 5,000,000 syzygy positions |
| Gambit dataset | EXISTS (mined pool) |
| Extractor v2 | BUILT + VALIDATED (300/300 config+men ground truth, 200/200 contact ground truth) |
| Depth-12 fleet eval | after dump |
| Matrix on full data | after evals |
| Fork slots | array plumbing BUILT (any N); router being finalized to this document |
| Per-expert .bins | after matrix + partition ruling |
| Training | nnue-pytorch, CUDA-optimized, 20GB VRAM approved for quality |
| Match | SPRT: expert-MoE vs monolithic, SAME TOTAL budget + 2x/4x/8x curves |

## 8. Open questions for ruling (nothing proceeds on these without the operator)

1. Tactics gate: threshold calibration method (ROC on puzzle-vs-quiet labeled sets) and the training-target design for puzzle positions (decisive-win labeling) — proposal before packing.
2. RESOLVED 2026-09-27: confrontation matching is RESIDUE-BASED (cancel common pieces,
   imbalance persists in ANY material context — "in many openings the mainline is BxN,
   which creates the N vs B imbalance"). Caveat recorded: residue conflates 2N vs N+B into
   N vs B (same difference multiset); if 2N vs N+B must stay distinct the predicate needs
   residue + on-board minors count.
3. RESOLVED 2026-09-27: balanced positions are SHARDED by the amount of locked pawns
   (operator ruling, repeated). Census distribution of center-locked files within balanced
   (partial data): lock_c=0: 53.4%, lock_c=1: 34.0%, lock_c=2: 9.8%, lock_c=3+: 2.7%.
   Shard cuts and count final at full-data census.
4. Starved-class folding (with mass numbers shown first).
5. Opening-boundary ply value (from data; UCI option).

RESOLVED by ruling 2026-09-27: mid-queenless REMOVED (imbalance buckets + balanced handle
queenless); tactics = one independent puzzle-trained expert outside the MoE (dual-signal,
tactic-first gate).

## 9. Laws (accumulated, binding)

- Gate verdicts by exit code, never grep; telemetry is the acceptance gate.
- No symlinks for stagepacks; pkill self-match law; game-boundary splitting only (never line-based).
- python-chess on rtx5090 is 0.31.4 (`board.result()`, not `board.outcome()`).
- Move field unsigned (`H`) in the bin struct.
- The failure identity is the retraining set; sampled 1.0 = noise — pass-firing gates must be
  statistically sufficient or exhaustive.
- Verify deliverables against the passed datasets, not proxies.
