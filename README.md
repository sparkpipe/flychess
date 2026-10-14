# flychess — the fly-brain chess project

Drosophila connectome (MaleCNS 188,778 neurons) learns chess.

## Architecture
- Features (color-blind, STM-relative) -> sensory/CX injection -> 2-step
  propagation through the real connectome -> 4096 move-slot readout
- FlyLeela engine: PUCT search over fly policy/value, menu-comparison value,
  opponent-trap layer, stance controller (nurture/chaos/desperation)
- Swarm mode: per-fly sensory dropout = diverse simulated flies

## Training: staged, gated curriculum (nothing advances below 100%)
1. piece movement instincts (open board)
2. two pieces: blocking / capture
3. three pieces: protection
4. king safety, pins, forks, discovered attacks
5. basic mates (KQvK, KRvK) -> 100%
6. K+P endings -> 100%
7. Dvoretsky, one chapter at a time

## Tools
- checklist.py — tournament-player gate (SF resistance, 100% bars)
- ladder_match.sh — Elo ladder vs calibrated anchors + SF UCI_Elo rungs

## dbnet — general labeling infrastructure

`tools/dbnet/` — label any FEN corpus with any engine/depth into a shared
DB via a fleet (cursor-protocol GET/POST sweep service, pooled on-demand
evals, deploy + telemetry). Engine-agnostic; see tools/dbnet/README.md
and tools/dbnet/EVALUATION.md (design post-mortem: three implementations
compared on complexity, debugging time, and throughput).

## tools/session — operational scripts (2026-10-11..14 era)

Everything that ran the pipeline, committed as-is. By category:

- **AJ corpus mining**: aj_worker.py, aj_launch2.sh, aj_audit.py,
  aj_materialize.py, aj_stats.py, fix_pgns.py — d10 labeling pipeline
  for the 6.2M-game postal/OTB corpus (490.9M positions)
- **Corrected nQ16 training**: db_export_correct16.py,
  verify_correct16.py, correct16.sh, correct16_round2.sh — the
  puzzle-label corruption fix + 16-expert retrain chain
- **General-net training**: build_general2.sh, train_general2.sh,
  build_general3_v2.py, train_general3_correct.sh — single-net
  training on clean data (general2 beats SF8 at 1s/move)
- **Clean-win filter**: find_clean_wins_spec.py, find_decisive_v2.py,
  audit_miniature_evals.py — operator-spec filter: rook advantage by
  move 20, no giveback below 2 pawns, no blunder-created wins
- **Match scripts**: match_*.sh — all engine wiring + fastchess
  invocations (1s, 10s, fixed-depth-20)
- **Corruption forensics**: compare_bins.py, pd_source_decomp.py,
  puzzle_hypothesis.py, expert_label_audit.py — the label-forensics
  chain that isolated the puzzle-label corruption
- **d20 fleet**: d20_insert.py, d20_setup.py, finish_d20.py,
  d20v2_launch.sh — fleet setup + finisher for the full-corpus
  d20 sweep (16.4M positions on 256 spark cores)
