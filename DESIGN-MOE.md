# PHASE-MoE DESIGN DOCUMENT — consolidated from the full operator record
**Version 1.6 — 2026-09-27. Single source of truth for expert taxonomy, routing, data
selection, and experiment protocol. Every line traces to an operator ruling; where a
reading was chosen it is marked [READING — ruling pending].**

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
- UCI: `EvalFile` (slot 1) + `EvalFile2..EvalFileN`; unfilled slots fall back to the first
  loaded net (never a zero-weights eval).
- Router = pure function of the position (material residue, pawn contact, total men, game
  ply). Cut points are UCI options calibrated from data, never frozen in code.
- Debug modes (`PHASE_MOE` env): 0=off, 1=route, 2=always-endgame, 3=dual-evaluate, 4=route trace.
- Trainer: official nnue-pytorch, features `HalfKAv2_hm^` (with caret), 40-byte nodchip
  bins (`<32shHHbB`, unsigned move field).
- Slot count is one constant; the plumbing carries any N.

---

## 3. THE EXPERT LIST

### 3.1 Structural experts

| # | Expert | Description | Routing predicate | Training data |
|---|--------|-------------|-------------------|---------------|
| 1 | **opening-gambit** | Early sacrificed or grabbed material; initiative-first play. | game_ply < opening bound AND asymmetric residue | existing gambit dataset (mined pool) + early-asymmetric segments |
| 2 | **balanced** (sharded) | Symmetric material residue, no listed confrontation. **Sharded by the amount of locked pawns** (ruling, §5): lock_c = 0 / 1 / 2 / 3+ | symmetric residue, no confrontation, past opening | balanced segments, sharded by lock_c at packing |
| 3 | **exchanges** (RESOLVED) | The transient moves of a trade: "the position is in one category, stuff happens to material balance, it settles down to the new material balance — the 'stuff happens' moves is the exchanges subset." Any move that changes which category a position is in and is not the stable end-state position. **These positions ALSO stay in the original material-balance training** (dual membership — the stable-category expert keeps them). | transient between stable material balances (category changed, not settled) | upward-slope positions during transitions, tagged `exch` by extractor v2 |
| 4 | **endgame-dvoretsky** | Practical endgames, 6-10 men, no listed confrontation. CONFIRMED. | 6 <= men <= 10 (after confrontation table) | dvoretsky-region segments |
| 5 | **endgame-tablebase** | Exact endgames, <=5 men. CONFIRMED. Syzygy training data COMPLETE (exactly 5,000,000 positions). | men <= 5 — absolute, first check | syzygy (done) |

### 3.2 Tactics expert (independent, puzzle-trained)

| # | Expert | Description |
|---|--------|-------------|
| 6 | **pure tactics** | ONE independent expert trained on the tactics puzzles (Lichess DB, 6,100,953 theme-tagged) — regardless of material balance, eval trajectory, phase. It only sees pure tactics. It exists OUTSIDE the MoE (dual-signal protocol below). Packing (RESOLVED): a puzzle is a set of moves that COMBINE into a winning combination — the ENTIRE combination is correct (made to be); secondary moves may also work — pack the full line. |

**Dual-signal player (ruling):** the structured-strategy MoE plays general-concept chess —
eval slope improvement, slowly squeezing to checkmate — while the tactics player cares only
about the immediate tactical win.

**Protocol (verbatim, assembled):**
1. **The tactics evaluator runs FIRST** — on every position, at evaluation time.
2. It **finds its best move** (the tactical candidate — the puzzle-trained net is
   move-producing, not just an eval).
3. Behind that move **there is a move tree at depth** — the tactical line. **The leaf
   node** is the ending position.
4. The **positional evaluator evaluates the leaf node** (depth = runtime parameter, UCI).
5. **If the positional evaluator says it is a big jump in win percentage — THAT is the
   trigger:** immediate tactical win confirmed, the tactics move is the answer. Done.
6. **No jump: no tactic** — normal positional MoE evaluation/routing.

**Tactical tree termination (ruling):** the tree expands **until the position is QUIET** —
no imminent trades, no other short-term tactics. Quiescence termination, not a fixed ply
count.

**Time management (ruling) — trajectory-driven:**
- **Investment phase:** early, we invest our time to get as good of a position as possible.
- **Harvest phase:** at some point we switch to "make reasonable moves quickly" — the hope
  is the time investment that got us into time trouble created a good position.
- **Critical-area rule:** if the position is LOSING win percentage and getting close to the
  critical area **40%**, we spend MORE time to try to get back to **45%+**.
- Constants given: 40% critical, 45% recovery. Investment→harvest switch: runtime parameter.
  Driver: the EMA'd win% trajectory over the game — same signal as the training bands.

TBD knobs: win%-jump size/reference for the trigger.

### 3.3 Material confrontation experts — 11 residue classes
(2B vs 2B REMOVED 2026-09-27 — "2B vs 2B is balanced! it is not an imbalance category";
2R vs 2R REMOVED 2026-09-27 — also symmetric, same ruling. Both flow into the balanced
shards by locked pawns. Symmetric material is never an imbalance category.)

**Matching (RESOLVED ruling):** RESIDUE-BASED — cancel common pieces; the imbalance is the
difference multiset and **persists in ANY material context** ("in many openings, the
mainline is BxN, which creates the N vs B imbalance" — an N-vs-B middlegame with queens and
rooks on is an N-vs-B position). Pawns are free (pawn-count differences are not
confrontations).

| # | Expert | Residue (either orientation) | Notes |
|---|--------|------------------------------|-------|
| 7 | **N vs B** | N v B | the BxN-mainline mass; also absorbs 2N vs N+B by residue (caveat below) |
| 8 | **2N vs N+B** | MERGED into N vs B (ruling 2026-09-27) | |
| 10 | **N vs R** | R v N | INCLUDES exchange-down play, both sides ("saving the draw being an exchange down is a super important skill" — the DOWN side's data comes from the both-sides dump) |
| 11 | **B vs R** | R v B | same, includes exchange-down |
| 12 | **R vs 2 minors** | R v {NN, NB, BB} — MERGED (ruling: "combine 2N vs R, 2B vs R with N+B vs R") | |
| 13 | **Q vs material** | Q v {RR, RN, RB} — MERGED (ruling: "combine R+B vs Q and R+N vs Q and 2R vs Q") | |
| 19 | **opposite bishops** | symmetric bishops, opposite majority square colors | square-color tag, fires in any material |

Exchange-down is NOT a separate expert — covered by N vs R / B vs R (ruling).
Starved classes fold into a neighbor ONLY on explicit ruling with mass numbers shown first.

---

## 4. Evaluation order and routing cascade

**Stage 0 — dual-signal tactics protocol** (§3.2): tactics evaluator first → best move →
tree until quiet → positional evaluates the leaf → win%-jump = tactic confirmed (done);
no jump → continue.

**MoE cascade (first match wins):**
1. men <= 5 → **tablebase** (exact knowledge is absolute)
2. confrontation residue table (§3.3) → its expert
3. men <= 10 → **dvoretsky** (general practical endgame, no confrontation)
4. equal-value capture available → **piece-trades**
5. game_ply < opening bound AND asymmetric residue → **opening-gambit**
6. symmetric residue → **balanced**, sharded by lock_c (§5)

Opening boundary ply: UCI option, calibrated from data. No other free constants.

---

## 5. Locked-pawn closedness — balanced positions only

Applies ONLY to balanced (symmetric-residue) positions. **Balanced is SHARDED by the
amount of locked pawns (ruling).** Census distribution of center-locked files within
balanced (partial data, 22% of pool): **lock_c=0: 53.4% · 1: 34.0% · 2: 9.8% · 3+: 2.7%**
→ shard cuts 0/1/2/3+, final at full-data census (3+ may fold into 2+ if starved).

Definitions (operator's, verbatim):
- **Locked**: facing pawns on the same file (white pawn directly below a black pawn).
- **Tension**: capturable pawns.
- **Open file**: no pawns at all on the file.
- **Edge files locked != closed** — center remains playable.
- **Before any contact**: neutral/undetermined.

Continuous metric (recorded on every position by extractor v2, available engine-side):
```
per file f in a..h:  weight w(f) = 2 for c,d,e,f; 1 for a,b,g,h
L = sum of w(f) over locked files;  T = pawn-attack tension instances;
O = sum of w(f) over fully-open files
L + T + O == 0  ->  NEUTRAL (no pawn contact yet)
else C = (L - O) / (L + T + O)   in [-1, +1]
```
Caution recorded: hard shard cuts on a continuum create eval discontinuity at boundaries —
sharding proceeds per ruling; C remains available as a smoothing/reporting axis.

---

## 6. Training data selection

- **Source**: `gambit/filtered.pgn` — decisive OTB games, winner >=2400, loser >=2000
  (~1.53M games). **BOTH SIDES' positions (ruling: "the losing side can also create
  training worthy data" — defensive play, exchange-down saving, is the DOWN side's data).**
  Expansion source for starved classes: the full 10.3M-game archive.
- **Evaluations**: depth 12 minimum (fleet, all 14 sparks). Depth-1 gambit evals are OBSOLETE.
- **Moves included**: castling, en passant, threefold repetition, underpromotions — all must appear.
- **Elo floor**: stay with 2400+ for our moves.
- **Segmentation** (extractor v2, built and validated: 300/300 config+men, 200/200 contact ground truth):
  - Blunder boundary: eval swing >= 120cp in cp space.
  - **Sub-division at every material change**; each sub-segment trains its own expert.
  - Exchange-move positions tagged `exch` (piece-trades data).
  - Per-position: config, opp/same bishops, lock_c/lock_e/tension/open, men, phase.
- **Trajectory classes** (W = 1/(1+exp(-cp/361))): press 55-70, convert 70-win, equalize
  45-55, defend 30-45, win 95+, collapse <30; segments classified start-band -> end-band.
  - **SELECTION RULE (verbatim rulings):** the result of the game does not matter — we look
    for **steady improvement of position**. "A 55% to 60% does not help us; a 40% to 55%
    does." THE FOUR CATEGORIES (ruling 2026-09-27): **30->45, 45->60, 55->70, 70->win**.
    "What is this 95%??? From 70% to win" — there is NO 95 wall; win begins at 70.
    Machine: pieces cut at band walls 30/45/55/70; credited when closing on a qualifying
    climb of EITHER side (white rising, or white falling = black rising); convert pieces
    (>=70) close at segment end while still climbing. Same-band positions are excluded.
  - Eval slope EMA'd (ruling).
- **The matrix**: (config x trajectory) with segments/ply counts — per-expert mass,
  cut-point calibration, starved-class detection.

### 6.1 Expert data census (partial data: 13,990,350 positions = 338K games = 22% of pool)

Balanced shard breakdown (total / qualifying / x4-qualifying projection):

| shard | total | share | qualifying | x4-qual proj |
|---|---|---|---|---|
| lock_c=0 | 1,789,470 | 52.8% | 257,387 | 1.03M |
| lock_c=1 | 1,163,281 | 34.4% | 195,361 | 781K |
| lock_c=2 | 338,001 | 10.0% | 63,833 | 255K |
| lock_c=3+ | 95,261 | 2.8% | 24,459 | 98K |

(The 3+ shard at ~98K filtered projection folds into 2+ unless the full-archive expansion
sustains it — per shard ruling, cuts confirmed 0/1/2/3+ and revisited at full census.)

Qualifying population (ANY band crossing = demonstrated improvement by either side —
downward in white-perspective wp = the black side improving; same-band excluded; partial
data, 22% of pool, winner-side decisive only — both-sides + full archive multiply further).
Gambit = early + ASYMMETRIC only (2,336 packed incl. dual-exchanges; the mined gambit pool
is essential for this expert). Exchanges (dual-membership bin): 383,942 packed on partial.
Packer: tools/pack_expert_bins.py, validated — all class counts match census exactly.

| expert | total positions | qualifying now | x4 proj (filtered) | x53.6 proj (both sides + full archive) |
|---|---|---|---|---|
| balanced (all shards) | 3,505,574 | 552,837 | 2.21M | ~29.6M |
| gambit | 3,041,084 | 538,777 | 2.16M | ~28.9M |
| N vs B (incl. 2N vs N+B) | 2,876,402 | 360,706 | 1.44M | ~19.3M |
| exchanges | 2,613,587 | 127,045 | 508K | ~6.8M |
| dvoretsky | 471,518 | 32,086 | 128K | ~1.7M |
| opp-bishops | 459,298 | 52,365 | 209K | ~2.8M |
| B vs R | 367,499 | 26,654 | 107K | ~1.4M |
| N vs R | 321,449 | 21,074 | 84K | ~1.1M |




| R vs 2 minors (merged) | 199,502 | 32,761 | 131K | ~1.76M |
| Q vs material (merged) | 114,000 | 16,735 | 67K | ~897K |
| 2B vs 2B | 6,453 | 891 | 3.6K | ~48K |

(2026-09-27 census-bug fix: N+B vs R / R+N vs Q / R+B vs Q previously read as zero —
residue keys were chess-ordered while lookups were alphabetically sorted; exactly the
three two-letter classes never matched. Operator caught it: "that is a VERY COMMON
occurrence." Engine-side router was never affected — it compares piece counts.)

External, in hand: tablebase 5.0M syzygy (done) · tactics 6,100,953 puzzles ·
gambit mined pool. Reference points: 14.6M positions -> ~2000-2200 Elo;
102M -> ~2400. Expansion plan: both-sides dump (RUNNING next, ~2x mass, adds the
defensive side), then full archive (~6.7x games) for starved classes; synthetic
from seeds ONLY on explicit ruling (self-play ban applies to real training).

---

## 7. Pipeline state

| Step | State |
|------|-------|
| Winner-only dump, filtered.pgn | RUNNING (~43M of ~63M) |
| **Both-sides dump** | tool built + smoke-tested; AUTO-LAUNCH armed on dump completion (~126M positions) |
| Depth-12 fleet eval | after both-sides dump |
| Extractor v2 | BUILT + VALIDATED |
| Expert census | DONE on partial (§6.1); rerun on full data |
| TB specialist | COMPLETE (5.0M) |
| Tactics puzzles | IN HAND (6.1M) — packing proposal pending (open question 1) |
| Fork slots | array plumbing BUILT (any N); router finalizes to this document |
| Per-expert .bins | after full census + shard-cut ruling |
| Training | nnue-pytorch, CUDA-optimized, 20GB VRAM approved |
| Match | SPRT: expert-MoE vs monolithic, SAME TOTAL budget + 2x/4x/8x curves |

### 6.2 Audit (2026-09-27, operator-ordered) — ACCEPTED NOISE

Random credited pieces verified with fresh depth-12 evals at every ply, independent
expert recomputation, bit-level round-trip:
- Round-trip 80/80 PASS; expert membership PASS (zero mismatches).
- Stored-vs-fresh evals: median |d| 8cp, p90 62cp, max 161cp; same-binary hash-size-only
  test flips 6.4% of positions across the walls — depth-12 search non-convergence
  ("chess is confusing" — operator). **RULING: acceptable noise.** Depth-12 continues;
  cp targets dominate training; category labels carry ~6% wall fuzz by design.
- Engine bug found+fixed during audit: fork segfaulted clearing accumulator caches of
  unloaded slots — only loaded slots are touched now.

## 8. Open questions for ruling

RESOLVED 2026-09-27: (1) puzzle packing = entire combination, all correct, secondary
solutions may also work; (2) exchanges = transient category-change moves, NOT routed away —
dual membership with the stable-category training; (3) 2N vs N+B merged into N vs B;
(4) balanced shards 0/1/2/3+ confirmed.

Still open:
1. Starved-class folding with mass numbers shown first (thinnest on full pool: qvmat 175K).

RESOLVED 2026-09-27:
- Opening boundary = **move 15 = 30 plies** (ruling); UCI default 30.
- Tactics-gate "big jump": NO separate knob — a jump is "big" iff it crosses at
  least one band wall (30/45/55/70). Same trajectory system, no new constant.

RESOLVED 2026-09-27: 2R vs 2R folded into balanced (symmetric, same as 2B vs 2B).

RESOLVED (record): mid-queenless REMOVED (imbalance + balanced cover it) · tactics = one
independent puzzle-trained expert, dual-signal protocol · confrontation matching =
residue-based · balanced sharded by locked pawns · both-sides data · draw-to-draw-only
exclusion · exchange-down covered by NvR/BvR.

## 9. Laws (accumulated, binding)

- Gate verdicts by exit code, never grep; telemetry is the acceptance gate.
- No symlinks for stagepacks; pkill self-match law; game-boundary splitting only (never line-based).
- python-chess on rtx5090 is 0.31.4 (`board.result()`, not `board.outcome()`).
- Move field unsigned (`H`) in the bin struct.
- The failure identity is the retraining set; sampled 1.0 = noise — pass-firing gates must
  be statistically sufficient or exhaustive.
- Verify deliverables against the passed datasets, not proxies.
