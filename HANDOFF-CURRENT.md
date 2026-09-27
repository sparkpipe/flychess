# FLYCHESS HANDOFF — 2026-09-27

## OPERATOR DIRECTIVES (standing, do not violate)

1. **NO UNAUTHORIZED EXPERIMENTS** — "do not do experiments I do not authorize"
2. **NO SHORTCUTS** — depth-1 evals, partial dumps, monolithic tests without asking: all violations
3. **NO LYING / SUGAR-COATING** — "STOP FUCKING LIE TO ME" / "DONT FUCKING LIE TO ME AGAIN"
4. **Be precise with numbers** — 380K/10.3M = 3.7%, not 20%
5. **The 8-specialist MoE is the goal** — material config × trajectory type
6. **Curated OTB data only for the real training** — not self-play

## THE PROGRAM (Path A: SF + our NNUE + phase-MoE)

**Goal:** World champion chess bot via Stockfish search + our phase-routed MoE evaluation.

**Architecture:** 8 specialist NNUE nets, each trained on a (material_config × eval_trajectory) cell of curated OTB data, loaded into a forked SF engine via EvalFile1-8, routed at eval time by piece count / material configuration.

## WHAT'S BUILT AND WORKING

### Stockfish fork (phase-moe-v0, branch on rtx5090)
- Location: `/home/spec/Stockfish/` (branch `phase-moe-v0`, commits thru `34332d9`)
- **Currently supports 2 slots only** (EvalFile + EvalFile2)
- Needs extension to 8 slots for the real experiment
- Routing: piece count ≤ 10 → endgame net (EvalFile2)
- **MUST extend to 8 slots with material-config predicates**
- Debug modes: `PHASE_MOE=0` (off), `1` (route), `2` (always-end), `3` (dual-evaluate)
- Bug fixed: per-slot `EvalFile` state (was zeroing net2 due to shared state)
- Bug fixed: per-slot `AccumulatorStack` (was corrupting evals)

### nnue-pytorch trainer (on rtx5090)
- Location: `/home/spec/nnue-pytorch/`
- C++ data loader built at `data_loader/cpp/build/libtraining_data_loader.so`
- Trainer works: `python3 train.py <data.bin> --max-epochs N --batch-size 1024`
- Serializer works: `python3 serialize.py <ckpt> <out.nnue> --features "HalfKAv2_hm^"`
- Deps installed (tyro, torch, etc. via `--break-system-packages`)
- **Feature set MUST be `HalfKAv2_hm^`** (with caret) to match engine architecture

### .bin data format packer
- Our packer in `tools/gen_smoke_data.py` / `tools/pack_curated.py`
- Correct 40-byte nodchip format (verified by C++ loader)
- PackedSfen uses Huffman coding (BitWriter class in each tool)

### Spark fleet infrastructure
- 14 sparks reachable via ssh from rtx5090 (spark1-8, sparka-f)
- SF built for ARM on each node (`~/extnvme/phase-moe/sf/src/stockfish`)
- Data gen workers deployable via `~/extnvme/phase-moe/launch.sh`
- **spark2 has the full infrastructure** (SF + python-chess + gen scripts)
- Other nodes need `python3 -m pip install --break-system-packages python-chess`

### Quick match tester
- `tools/quick_match.py` — 3-game matches vs SF at configurable skill level
- Uses python-chess 0.31.4 (old API — `board.result()` not `board.outcome()`)
- Runs on rtx5090 CPU, ~90s per 3-game match at 0.5s/move

## NETS TRAINED SO FAR (all on rtx5090)

| net | data | arch | Elo estimate | file |
|-----|------|------|--------------|------|
| self-play baseline | 102M self-play depth-10 | monolithic | ~2400 | `our_net_v2.nnue` |
| self-play 7-way (7 nets) | same 102M, piece-count sorted | 7 nets | untested | `nnue_7way/bucket_{0-6}.nnue` |
| curated monolithic | 14.6M OTB (partial dump, ~3.7% of archive) | monolithic | ~2000-2200 | `curated_net.nnue` |

**None of these are the operator's design.** The real design needs:
- Full OTB dump (all ~1.53M qualifying games)
- Deep evaluations at depth 12+ on the full dump
- Trajectory segment extraction on full data
- Material-config bucketing
- 8 specialist nets trained on their cells

## MATCH RESULTS (3-game spot tests)

### Self-play baseline (our_net_v2.nnue)
| skill | Elo | score |
|-------|-----|-------|
| 8 | ~1600 | 1.00 |
| 12 | ~2000 | 1.00 |
| 15 | ~2300 | 0.67 |
| 18 | ~2600 | 0.50 |
| 20 | ~2900+ | 0.33 |

### Self-play 7-way with endgame routing (bucket_6 in EvalFile2)
| skill | score |
|-------|-------|
| 8 | 1.00 |
| 12 | 1.00 |
| 15 | 0.50 |
| 18 | 0.33 |
| 20 | 0.17 |

### Curated monolithic (partial data)
| skill | score |
|-------|-------|
| 8 | 1.00 |
| 12 | 0.83 |
| 15 | 0.67 |
| 18 | 0.17 |
| 20 | 0.00 |

**Note:** All these are 3-game samples — statistically weak. The 7-way and curated results are from partial/incorrect data and do NOT represent the operator's design.

## DATA ASSETS ON rtx5090

| asset | path | size | status |
|-------|------|------|--------|
| OTB game archive | `/home/spec/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn` | 8.6GB | complete |
| Prefiltered PGN (decisive, 2400+/2000+) | `/home/spec/chess-lab/gambit/filtered.pgn` | ~6GB | complete |
| Partial OTB dump (winner positions) | `/home/spec/chess-lab/otb_all_positions.txt` | 15.7M positions | ~3.7% of archive |
| Deep evals on partial dump (depth 12) | `/home/spec/chess-lab/otb_evals/combined.txt` | 15.7M evals | covers the partial dump |
| Trajectory segments (partial) | `/home/spec/chess-lab/otb_segments.jsonl` | 15.7M positions tagged | covers the partial dump |
| Curated training .bin (partial) | `/home/spec/chess-lab/curated_training.bin` | 14.6M positions | draw-to-draw excluded |
| GAMBIT pool (mined) | `/home/spec/chess-lab/tbpools/GAMBIT.jsonl` | 40K positions | from 94K matched games |
| DEGM2 corpus | `/home/spec/chess-lab/tbpools/DEGM2_Ch*.jsonl` | 21K positions | book-audited |
| TB specialist data (generating) | `/home/spec/chess-lab/tb_training.bin` | 3.1M/5M | syzygy ≤5 pieces |
| Self-play fleet data | `/home/spec/chess-lab/fleet_data_all.bin` | 102M positions | depth-10 self-play |
| Lichess puzzle DB | `/home/spec/chess-lab/puzzles/lichess_db_puzzle.csv` | 6.1M puzzles | theme-tagged |
| Syzygy 3-4-5 | `/home/spec/syzygy/` | 926MB | WDL+DTZ |
| 6-piece WDL download | spark2:~/extnvme/phase-moe/syzygy6/ | checking | from lichess |

## CRITICAL PENDING WORK (in order)

### 1. Complete the OTB dump
- The single-threaded dump processed only ~3.7% of qualifying games
- **Use `gambit/filtered.pgn`** as input (already prefiltered to decisive + 2400+/2000+)
- Expected yield: ~63M winner positions from ~1.53M games
- Tool: `tools/otb_dump.py` with `PGN=gambit/filtered.pgn`
- **IMPORTANT: the dump script has no resume. Must run to completion.**
- **DO NOT** use the depth-1 evals from the gambit mining — they are OBSOLETE
- **DO NOT** use line-based PGN splitting — it breaks games at shard boundaries

### 2. Deep-evaluate the full dump on spark fleet
- Deploy full position dump to all 14 sparks
- Evaluate at depth 12 (NOT depth 1, NOT depth 10 — depth 12 minimum)
- Tool: `tools/otb_eval_fleet.py` on each spark
- Expected time: ~30-60 minutes fleet-wide for 63M positions
- Each spark needs the position file + the eval script + SF binary

### 3. Run trajectory segment extraction on full data
- Tool: `tools/segment_extractor.py`
- Game boundary detection: ply decrease (NOT ply==1)
- Fix `None` cp values: `parts[7] != 'None'` guard
- Excludes draw-to-draw trajectories (operator ruling)
- Produces (trajectory_type, material_config) tagged positions
- Stats tool: `tools/matrix_report.py`

### 4. Extend the SF fork from 2 slots to 8 slots
- Currently: `EvalFile` + `EvalFile2` with piece-count routing
- Need: `EvalFile1-8` with material-config routing
- Same pattern as v0: per-slot Network, per-slot EvalFile, per-slot AccumulatorStack
- Files to modify: `engine.h`, `engine.cpp`, `search.h`, `search.cpp`
- Routing predicates from operator's design (see below)
- This is C++ work in `/home/spec/Stockfish/src/`

### 5. Build per-specialist .bin training files
- From the trajectory×config matrix, group positions by specialist assignment
- Pack each specialist's positions into .bin format
- Tool: variant of `pack_curated.py` with config-based filtering

### 6. Train 8 specialist nets
- Each specialist trains on its own .bin
- Same nnue-pytorch trainer
- Budget: operator decides (they predicted MoE needs more training time)

### 7. Load into fork, SPRT match
- 8 EvalFile options → 8 nets
- SPRT vs monolithic baseline at equal total training budget
- Also run at 2×, 4×, 8× budget for the curve comparison

## OPERATOR'S SPECIALIST DESIGN (do not deviate)

**8 specialists:**
0. opening-normal
1. opening-gambit
2. mid-positional
3. mid-open
4. mid-queenless
5. mid-tactics
6. endgame-dvoretsky (practical endgames, 6-10 pieces)
7. endgame-tablebase (exact endgames, ≤5 pieces, syzygy)

**Material configs for routing (from operator):**
- N vs B, 2N vs N+B, 2B vs 2B, 2N vs R, N+B vs R, 2B vs R, N vs R, B vs R
- 2R vs 2R, 2R vs Q, R+N vs Q, R+B vs Q
- Bishops of opposite colors (specific drawing/technique patterns)
- Exchange-down positions (R for N/B — how to hold the draw)
- **"saving the draw being an exchange down is a super important skill"**

**Open/closed is a SPECTRUM based on pawn contact:**
- Locked: facing pawns on same file (e4 vs e5)
- Tension: capturable pawns (important factor)
- Open file: no pawns at all
- Edge files locked ≠ closed (center still playable)
- Before pawns come in contact: neutral/undetermined

**Eval trajectory types (from operator):**
- press: 55% → 70% (pressing small advantage)
- convert: 70% → win (finishing)
- equalize: 45% → 55% (from worse to balanced)
- defend: 30% → 45% (fighting back from clearly worse)
- **Exclude: draw-to-draw** (no skill demonstrated)
- **Include segments from losing games** (improving play is valuable regardless)
- One mistake starts a new segment (blunder = boundary)

**Segment sub-division within material changes:**
- If material config changes mid-segment, sub-divide
- Each sub-segment trains the appropriate specialist
- The exchange moves themselves → piece-exchange specialist
- "The important thing is to capture the full sweep of the winning moves"

## LAWS AND RULES (accumulated)

- Exhaustive gates for verification; sampled gates for telemetry only
- Best-snapshot + answer-set (verification, not load-bearing selector)
- PR flow: test via PR, merge at stability
- Pause/resume must be smooth (checkpoint everything)
- Sparks: 10-15GB RAM limit (20GB approved for quality), nice-10, NVMe-local
- No symlinks for stagepacks
- `pkill` self-match law: never combine kill patterns and launch text in one ssh
- python-chess on sparks is 0.31.4 (old API — no `board.outcome()`)
- Stockfish on sparks is ARM (GB10) — x86 binaries won't run
- SF arch hash: use `--features "HalfKAv2_hm^"` (with caret) for the trainer
- Struct format for .bin: `<32shHHbB` (H for move, not h — overflow)

## RUNNING PROCESSES (as of handoff)

- **OTB dump on prefiltered PGN**: may be running from `gambit/filtered.pgn` → `otb_complete_dump.txt` (check with `pgrep -f otb_dump`)
- **TB specialist generation**: 3.1M/5M positions (`tb_gen.out`)
- **Spark fleet self-play workers**: may still be running on some nodes
- Check and stop unauthorized processes before starting new work

## KEY FILES ON rtx5090

```
/home/spec/Stockfish/          # SF fork (branch phase-moe-v0)
/home/spec/nnue-pytorch/       # NNUE trainer
/home/spec/chess-lab/           # main workspace
  games/LumbrasGigaBase_OTB_Complete.pgn   # 8.6GB OTB archive
  gambit/filtered.pgn                       # prefiltered (decisive, 2400+/2000+)
  gambit/evals.txt                          # depth-1 evals (OBSOLETE — do not use)
  otb_all_positions.txt                     # partial dump (15.7M from 3.7% of games)
  otb_evals/combined.txt                    # depth-12 evals on partial dump
  otb_segments.jsonl                        # trajectory segments on partial dump
  curated_training.bin                      # packed curated data (partial)
  our_net_v2.nnue                           # self-play baseline net
  curated_net.nnue                          # curated monolithic net (partial data)
  nnue_7way/bucket_{0-6}.nnue              # self-play 7-way nets
  tb_training.bin                           # TB specialist data (generating)
  fleet_data_all.bin                        # 102M self-play positions
/home/spec/syzygy/                          # 3-4-5 piece tablebases
```

## WHAT NOT TO DO

1. Do NOT train monolithic nets and call them the experiment
2. Do NOT use depth-1 evaluations for anything
3. Do NOT use self-play data for the curated training
4. Do NOT run experiments without operator authorization
5. Do NOT claim partial data is complete
6. Do NOT use line-based PGN splitting (breaks games)
7. Do NOT use `gambit/evals.txt` (depth-1, obsolete)
8. Do NOT test the 2-slot fork and call it the MoE comparison
