# FLYCHESS HANDOFF — runtime state for the next session (2026-09-19 ~21:00 UTC)

*Read DESIGN.md first (repo root + ~/chess-lab/DESIGN.md on rtx5090).
This doc = RUNTIME state, open threads, and the operator's NEXT DIRECTIVE.*

## Where the program is (one paragraph)

The fly's movement foundation is complete (v3+ANORM, 27/27 sweep, mate1 1.0,
9/13 TB families — PR #1 merged; weights release `milestone-weights-2026-09-19`
live; PR #2 open with everything since). Dvoretsky-as-monolith hit a measured
wall: every shared-substrate run plateaus 0.55–0.68 on the book chapters
(loaded fly REGRESSING by the progress metric; scratch fly, H01 graft
[1mm³ human cortex, 200K-node snowball], and MoE fly all plateau in the same
band; MoE's per-chapter peaks were highest [Ch2 0.877]). Diagnostics proved
the wall is interference, not difficulty: ONE question alone = **50 steps to
stable** (measured: /tmp/one_question.jsonl, tools/one_question.py, RESULT
STABLE first_pass_step=50); the saturation census showed all tasks share the
same top-20K neurons (97% overlap); basin analysis showed the connectome is a
global mixer (no territorial specialization possible); probes showed the frozen
activations lack the book signal (0.50–0.55 ceiling). The winning architecture
per the operator: **MoF — mixture of flies** (dedicated fly per specialty,
layered like human chess learning: basics automatic → next layer reads them
as blackboxes → superfly on top). Layer-0 = dedicated flies per section
(book split into position-clustered sections).

## THE OPERATOR'S NEXT DIRECTIVE (start here)

1. **Measure single-problem solve iterations at n=10**: train 10 separate
   one-question flies (10 different DEGM questions), record steps-to-stable
   for each. (tools/one_question.py is the harness — generalize it to take
   a question index; ONE_Q pool+pre pattern is the template; ~50 steps each
   expected from the first measurement.)
2. **Solve ALL problems in isolation** — one dedicated fly per question
   across the whole corpus (1601 DEGM questions + the 9 TB pools + mate
   pools as the operator extends scope). Each is ~50 steps ⇒ the full DEGM
   corpus ≈ 80K steps total ≈ ~12h serial on rtx5090, trivially
   parallelizable (but sparks NOT approved — rtx5090 only for now).
3. **The weight-overlap clustering (operator's idea)**: each solved question
   yields a weight vector (its fly's trained deltas — W_sens + theta + gains,
   NOT the frozen WT). Group questions by HIGH weight overlap (similar
   problems — small changes solve both, mergeable into one fly) and LOW
   overlap (lots of separate capacity — keep separate or pair for coverage).
   This replaces/augments the feature-space section split: the grouping
   becomes measured from solutions, not guessed from inputs. Deliverable:
   the overlap matrix + clusters; it feeds layer-1 reader grouping.

## Live processes (rtx5090)

| process | state | notes |
|---|---|---|
| ch4 proof (tools/mof_ch4_serial.py) | RUNNING | s1 (60 pos): NEW-BEST 0.6833@500, 25/60 solved recorded at peak; outcome-split + answer-set snapshots armed; caps at 12K steps |
| everything else | STOPPED per operator | MoE flies, H01 graft, scratch, v3_s6 trainer all parked |

## The one-question result (the calibration constant)

ONE question (2-move choice, Kg8 draws/Ke7 loses): **first pass at step 50,
stable through 10 gates by step 500** (16K board-visits). Harness:
tools/one_question.py + tools/stage_one_q.sh (writes pool + pre.npz; the
gate MUST have _pre arrays — rows without them score a silent 0/0).

## Key laws earned this session (full detail in memory + git log)

- The gate is EXHAUSTIVE, every block, always (operator ruling; sampled
  telemetry is dead). Pass claims additionally verified by full sweep.
- Best-snapshot + answer-set: track the peak exhaustive score, snapshot the
  model AND record the solved-position set AT the peak; splits use the
  recorded set (re-eval is verification only, never load-bearing).
- Outcome-driven splitting: solved → own subsection; failed → residual.
  No arbitrary halving.
- pkill SELF-MATCH: never combine kill patterns and launch text in one ssh
  cmdline (killed its own shell twice this session). Kill in its own ssh,
  launch in its own ssh, verify separately.
- Precomputed .pre.npz arrays serve BOTH gate and training batches now
  (train/playback aligned); rows without _pre silently break the gate.
- Progress metric: tools/progress_metric.py (deficit-to-goal trend +
  up/down ratio). Use it for any run.
- The DEGM generator (tools/degm_pools.py) is nondeterministic at the
  margin (1600 vs 1601 rows across runs — SF grader variance). The repo
  copy is the source of truth; regenerate consciously.
- Sparks: NOT approved for compute. Estimate only: ~6× slower per node
  (bandwidth-bound), ~2.5× fleet aggregate win.

## Artifacts & repos

- Repo: sparkpipe/flychess, branch `dvoretsky-lookahead-2026-09-18` (PR #2
  open; PAT with PR write is embedded in the remote URL — use
  `TOK=$(git remote get-url origin | sed ...)` for gh).
- DEGM question pools + book PGN: committed to the repo (tbpools/DEGM_Ch*.jsonl,
  DEGM.pgn). TB/mate pools node-only (regenerable).
- Section pools: tbpools/DEGM_Ch*_s*.jsonl + .pre.npz on the node (44
  sections from the drifted file; the REGENERATED split differs — Ch4 now
  = one 60-row section s1; resplit+precompute done on clean data).
- H01: 63M-edge human connectome extracted and verified (edgesV5_part*.npz
  in ~/chess-lab/h01_edges/; pre/post/typ arrays; tools/h01_edges_byid.py).
  The graft experiment plateaued like the rest — parked, substrate available.
- Checkpoints on node: fly_cb_v3_s6.pt (movement+TB9+DEGM-partial),
  fly_degmscratch.pt, fly_h01_degm.pt, fly_moe_degm.pt, fly_moe_mix.pt,
  flies_mof/l0_*.pt (section flies).

## Standing rulings to honor

- No fundamental training changes without operator permission FIRST.
- No spark compute without approval. rtx5090 GPU is shared with a foreign
  farm_server (port 7793) — never touch it.
- Timer/automations: ALL deleted (operator found passive monitoring
  useless). Act, don't watch.
- Honesty ledger: retracted claims stay retracted (the circular
  set-margin "confidence" result; "effectively exhaustive"). The gate
  dtz-None guard, the always-exhaustive ruling, and the answer-set
  snapshots came from operator corrections — read the git log.
