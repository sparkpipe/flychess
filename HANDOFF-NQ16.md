# HANDOFF — nQ16 era (2026-10-06, session near compaction)

## READ THIS FIRST
Operator directive if n16 engine misbehaves: **start from the nQ13 codebase** (proven working:
`/mnt/cold-raid6/chess-audit/nq13_binary_backup` binary; sources in repo `chess-lab/tools/nq16/nq13-engine/`
+ full working tree at `/srv/workspace/flychess/src/Stockfish-act` for 23-slot or the 13-slot fork tree),
**replace the 13 experts with the top 13 of the 16 experts, change the router to those 13 only**.
Minimal changes to a working engine. Do NOT port engine structure again from scratch.

## Current instant status (2026-10-06 ~13:00)
- **n16.0 16-slot engine BUILDS and mostly WORKS**: `/srv/workspace/flychess/src/Stockfish/src/stockfish`
  (rebuilt from the sf8 13-slot proven base + my 16-slot port).
  - 16 EvalFile options emitted ✓
  - route command verified: startpos→slot11 op_even_l0, KRK→slot0 tb, imbalanced QGD→slot5 piece_down ✓
  - all 16 nets load (after fixing resize_threads passing only 13 network pointers — THE segfault cause)
  - `bench 16 1 8 default depth` SEARCHES CORRECTLY (24837 nodes, real PVs) with run16 nets
  - REMAINING QUIRK: piped `go movetime N` returns instantly (depth 1, nodes 0, bestmove a2a3) —
    bench and depth-limited go work; movetime-in-pipe does not. NOT yet root-caused. The saved
    nq13 backup binary shows the same quirk in pipes but worked fine under fastchess (which uses
    `go` differently). VERIFY with a real fastchess match before concluding the engine is broken.
- 16 nets at `/srv/workspace/chess-active/engine16/` (epoch-picked best-val per expert, listed below).

## The 16-expert taxonomy (operator-approved, EXPERTS16.md in repo)
Order: tb(0) mvr(1) rv2m(2) qvmat(3) nvb(4, incl 2Nv2B) piece_down(5) oppb(6) dv_Q(7) dv_R(8)
dv_rest(9) op_pawnimb(10, fm<15 pawn-imbalance) op_even_l0/l1/l2+(11-13) mg_unsafe(14) mg_safe(15).
Rules: no catch-all (piece_down catches all unnamed asym), no side-to-move keys, castling is
first-class key, dv AFTER all imbalance checks and oppb (operator: "we moved dv after oppb for a reason").
Python reference router: repo `chess-lab/tools/nq16/routerB.py` (sources: /tmp/routerB.py on box).

## Best-val nets picked (engine16/) — epoch / val_corr
op_even_l1 e9 0.573 (box run), op_even_l0 e37 0.569, op_even_l2p e474 0.538, op_pawnimb e252 0.505,
mg_unsafe e489 0.342, mg_safe e~1 0.334 (restarted, low), piece_down e65 0.275, oppb e2 0.284,
dv_R e126 0.280, mvr e2 0.264, dv_Q e0 0.250 (early), nvb e90 0.254, dv_rest e582 0.262,
qvmat e418 0.177, rv2m e334 0.236, tb e51 0.218 (all still training; re-pick later for n16.1).

## MAIN nQ16 training: STOPPED 2026-10-06 (operator order)
- Verdict at stop: NO expert still learning — all 16 peaked epochs ago, tails flat/declining.
  Final bests: op_even_l1 0.573, op_even_l0 0.576(e44 NEW), op_even_l2p 0.538, op_pawnimb 0.505,
  mg_unsafe 0.342, mg_safe 0.336(e12 NEW), piece_down 0.275, oppb 0.284, dv_R 0.280, mvr 0.263,
  nvb 0.254, dv_Q 0.250, dv_rest 0.262, qvmat 0.177, rv2m 0.236, tb 0.220(e65 NEW, serialized from ckpt).
- engine16/ re-picked with the 3 improvements (op_even_l0 e44, mg_safe e12, tb e65) = n16.1 candidate set.
- Main nets+ckpts backed up on cold RAID `/mnt/cold-raid6/chess-audit/train16_backup/`; stale
  epoch ckpt piles deleted (spark0/1/7). Old scripts kept in repo `chess-lab/tools/nq16/anti16/`.

## 2026-10-07: ANTI STOPPED + SPARKS FREED + nQ-vs-n16 MATCH DONE
- Anti verdict: done — every spark expert peaked long ago (bests at e50-e2444 of much longer
  runs). Box op_even_l1 stopped too (best e87 corr 0.8384). 16 best anti nets at
  `/srv/workspace/chess-active/engine16anti/` (tb .456, mvr .579, rv2m .588, qvmat .490,
  nvb .684, piece_down .456, oppb .688, dv_Q .664, dv_R .687, dv_rest .675, op_pawnimb .799,
  op_even_l0 .836, op_even_l1 .838, op_even_l2p .869, mg_unsafe .693, mg_safe .711).
- ALL sparks freed for the other dev (kill script `chess-lab/tools/nq16/match16/free_sparks.sh`);
  census-verified only sparkpipe_weightd remains (their serving, do not touch).
- d20 fleet FINISHED (253430/253430 everywhere); outputs collected to
  `/mnt/cold-raid6/chess-audit/depth_db/shards/d20_spark*.tsv` (3.8M lines) — d12→d20 delta
  mining now unblocked. Merge collector still TODO.
- Box disk hit 100% (runanti ckpts 164G + run16 leftovers 98G — box anti watcher died at
  startup because checkpoints/ didn't exist 2s after launch). Cleaned to 258G free.
  Also found+killed 23-era zombie trainer (pd_down off cold RAID!) + 2 train23_queue workers.
  Crontab now EMPTY. LESSON: watcher must wait for checkpoints dir, not exit.
- NPS (fixed-node 500k, 6-position mix, 1 thread): nQ13 ~473k, n16 ~416k → ratio 1.137
  (16-slot switching cost ~14%, vs nQ.23's 2×). n16 = engine16 nets (with op_even_l0 e44,
  mg_safe e12, tb e65 re-picks).
- MATCH (6 games, 3open.epd, color-reversed, nQ st=1 vs n16 st=1.14 node-calibrated):
  **nQ 6-0 n16** — all decisive, nQ won both colors on all 3 openings. Engine mechanically
  perfect (60-90 ply games, exact time compliance, no forfeits — the piped-movetime quirk is
  confirmed a pipe artifact). Gap is NET quality, not engine code. PGN+log:
  `/srv/workspace/chess-active/matches/nq_vs_n16_1s.{pgn,log}`.
- nQ13 wiring (tourney6-proven): EvalFile=balanced_l0 EvalFile5=nvb (slot nvsb!) EvalFile13=tactics (slot tb!).
- Live page feed REBUILT: box `/srv/workspace/chess-active/matches/` runs `match_feed.py`
  (PGN→games.json) + 3× `match_eval_worker.py i 3` (SF17 multipv-5 d12/20/25, flock'd).
  Mac serve.py repointed (was dead /extnvme path), title refreshed. Page verified in real
  browser: 6 games, board, movelist, evals filling.

## ANTI16 training (STOPPED 2026-10-07 — see above)
- Same spark→expert map, run dirs `~/runanti_<expert>/` on sparks,
  `/srv/workspace/chess-active/runanti_op_even_l1/` on box. Bins: `~/anti_<e>.{train,val}.bin`
  on sparks (renamed from anti16/<e>.*.bin on box). skip=3, no time limit, batch 4096,
  epoch-size = full bin, val-size = min(30000, val records).
- Watcher everywhere: `/tmp/anti_watch_spark.sh` (sparks) / `/tmp/train_watch_generic.sh` (box),
  hourly wallclock ckpt keep-3, per-epoch .nnue export + ckpt delete.
- Keepalive cron (30min): `/tmp/fleet_keepalive_anti.sh` — prunes nets, backs up to
  `/mnt/cold-raid6/chess-audit/anti16_backup/`, RESTARTS dead watchers, REPORTS dead trainers
  (no auto trainer restart). Log: /tmp/fleet_keepalive.log (grep "anti cycle").
- Fixed nnue.py (true-cp val_corr/val_mae/val_median) pushed to ALL sparks
  (`~/nnue-pytorch/model/nnue.py`, original kept as nnue.py.bak_main16).
- Early signal: anti tb val_corr 0.33 by e31 (main tb best was 0.220) — anti bins are cleaner.
- Ops scripts committed: `chess-lab/tools/nq16/anti16/` (kill16main, cur16, pick16,
  anti_watch_spark, launch_anti, distribute_anti16, fleet_keepalive_anti).

## d20 fleet (running)
All 16 sparks × 16 cores sf17_arm depth-20. Done ~3.47M/7.2M at resume; worker `~/d20_spark_worker.py`
per spark reading `~/d20job.txt` appending `~/d20out.tsv`. Collector not yet written — when done,
merge outputs into `/mnt/cold-raid6/chess-audit/depth_db/shards/` (format fen<TAB>cp stm-pov).
Use: d12→d20 delta segments per operator design (add d20 rows to d12 corpus; corrected labels for
invalidated segments = "don't do this"; offset-tolerant matching).

## Anti engine
`anti16` bins rebuilding via `/tmp/assemble16anti2.py` (log /tmp/assemble16anti2.log) —
main+anti overlap measured at 47.9% (3.78M shared FENs; band-overlap duplication, identical labels;
operator informed, no dedupe ruling yet). Anti training NOT started (planned: parallel on same GPUs
once bins done — GPUs ~50% idle from ckpt-write overhead).

## PR / repo (RULE: all code via PR to github.com/sparkpipe/flychess, merge when working)
- PR #3 branch `nq16-era`: 96 files — all pipeline scripts `chess-lab/tools/nq16/`,
  ui/ (serve.py index.html pieces.js), nq13-engine/ preserved sources, engine patches.
  Second commit added nq13-engine. PAT in `~/flychess/.env` (Mac) — remote URL already has token.
  Push needs `-c credential.helper=` (keychain stale). gh CLI needs GH_TOKEN=<pat>.
  **NOT YET IN PR: the 16-slot engine port files** (phase_moe16.h, engine.cpp 16-slot,
  resize_threads fix, uci names16) — COMMIT THESE to nq16-era next.

## Key paths
- Active tier (NVMe /extnvme DIED ext4 emergency_ro — replace disk): now `/srv/workspace/chess-active/`
- train16 bins: `/srv/workspace/chess-active/train16/` (16 experts, incl 43.5M miniature rows,
  8.69M puzzles, DEGM, syzygy; salvaged labels `/mnt/cold-raid6/chess-audit/salvage/labels/`)
- Corpus/segments: `/mnt/cold-raid6/chess-audit/wp_fit/segments/` (+ /extnvme/segments copy lost)
- Old nets: engine23/ (23-slot), nets/ (13-slot nQ) on cold-raid6.
- Live games page: Mac `~/games-live/` (serve.py on :8077; feeds from box /extnvme DIED — page frozen
  at last cache; rebuild exporter+worker on chess-active when needed).
- Match tooling: `/extnvme/active/run_match.sh` LOST; recreate from PR `night_launcher` patterns +
  fastchess syntax in this doc below.

## fastchess PROVEN invocation (from tourney7.sh)
```
$FC -engine name=A cmd=PATH option.EvalFile=... [option.EvalFileN=...] [st=N] \
    -engine name=B cmd=PATH ... \
    -each proto=uci st=10 timemargin=200 -rounds 3 -repeat -recover \
    -openings file=3open.epd format=epd -pgnout file=OUT.pgn
```
3open.epd on cold-raid6/chess-audit (survives) + chess-active copy. sf8=/usr/games/stockfish;
SF17 x86 at /srv/workspace/flychess/src/sf17/src/stockfish (for UI evals).

## Standing operator rulings (do not re-litigate)
- 16-expert taxonomy EXACTLY as ordered above (merge dv after oppb; qvmat covers Q v 2-3 pieces;
  nvb absorbs n2v2; mg my/other merged; piece_down; no pd_up/pd_down; op_pawnimb fm<15; no
  undeveloped gate; oppb = exactly 1Bv1B opposite colors — verified 100%).
- random_fen_skipping N is odds-style N/(N+1); use 3 (75%) — operator confirmed "skip 3".
- Checkpoints: .nnue every epoch (export then DELETE 1.9GB ckpt), full ckpt hourly only
  (CHECKPOINT_INTERVAL_SEC=3600; 16×2GB/hr OK; NOT every 10 min).
- Miniatures: every winner-side position move 1→finish, NO SF grading (d16 grading was wrong).
- Draw games stay in training. Labels: fleet sf17 d12 cp (white-pov? = winner-side per segments;
  NOTE unresolved label-perspective question — training works, corr climbs, don't touch).
- Median metric required alongside mean/corr.
- "why is X slow/broken" answers: 256 cores for label bursts; hourly ckpts; PRs for all code.
- nQ13 backup binary + sources preserved (RAID + repo) — do not lose.

## Session war stories (avoid repeats)
- SSH self-match kill: never pkill -f with strings appearing in your own cmdline; pkill -x or
  exact pid. This bit me 4+ times.
- Shell heredoc through ssh mangles quotes — write script files locally, scp, run.
- sed/python patch engines: ALWAYS brace-aware or full-region replace; regex `{[^}]*}` truncates
  at inner braces (caused every engine.cpp corruption today).
- The 16-slot port's real bugs were: resize_threads passing 13 network pointers (segfault slots
  14-16), stale EX_DVORETSKY in search.cpp, PhaseMoeNames13 vs Names. ALL FIXED in current tree.
- sparkc GPU is FINE (device asserts were corrupt-checkpoint resume, not hardware); it hosts
  sparkpipe_weightd (serving, don't kill); use sparkc CPU/GPU freely.
- Disks: sparks filled by 1.9GB×every-epoch ckpts twice. Watchers + keepalive now bound it.
- /tmp on box is tmpfs? NO — /tmp persisted across days; but PR rule exists because /tmp+extnvme
  both proved mortal. spark3 has stray 832GB ~/extnvme/salvage (old, not mine — left for operator).

## Immediate next steps (in order)
1. Commit 16-slot engine port files to nq16-era PR branch.
2. Verify n16.0 with a REAL fastchess match (1s vs nQ13 backup binary) — piped-movetime quirk
   may be test artifact only. If engine misbehaves in real match → operator fallback: nQ13 base
   + top-13-of-16 (drop qvmat? tb? mg duplication? — operator decides or propose by corr rank:
   drop the 3 lowest-corr = qvmat, tb, mg_safe or dv_Q-e0).
3. d20 collector + merge when fleet finishes (~3.7M remaining).
4. anti16 bins → launch anti training in parallel (GPUs half-idle).
5. Re-pick best nets (runs still improving) → n16.1 when operator says.
6. Morning list: sparkc fine (no reboot needed), NVMe replacement, tb/qvmat label autopsies,
   provenance+audit re-run on rebuilt bins, live-page feeds onto chess-active.

## 2026-10-07 LATER: FULL TEARDOWN (operator order: disk cleanup + no zombies + all code in repo)
- Nets fully backed to cold RAID first (train16_backup + anti16_backup, ALL spark nets dirs;
  tb/mvr/dv_Q have only 1 main net each — their watchers died mid-run, picks live in engine16/).
- Sparks cleaned: ckpts, nets, bin copies, d20 files, ALL /tmp launch+watch scripts deleted.
  Freed 11-20 GB/spark (~260 GB fleet). Kept: metrics.csv + train.log per run dir (autopsy value).
- sparkc had a live main-era watcher loop (train_watch_spark.sh) — killed + script removed.
- Box cleaned: run16/runanti op_even_l1 ckpts+nets, /tmp junk nets, ALL trainer-launch scripts
  (archived to repo first). /srv/workspace at 260G free. Box crontab EMPTY. All spark crontabs EMPTY.
- Zombie census CLEAN fleet-wide (train.py/train23_queue/watchers/launchers/queues/keepers: zero).
- ALL code in repo: chess-lab/tools/nq16/box-tmp/ = 170 scripts archived from box /tmp
  (engine port/patch series, diagnostics, assemblers, launchers); ui/pieces/ glyph assets added;
  match16/ has audit/clean/census/backup scripts. Branch nq16-era pushed.
- NOT deleted (not mine): spark3 ~/extnvme/salvage = 832 GB (from-spark6, phase-moe rescue copy).
  Surfaced to operator — their call.

## 2026-10-07 NIGHT: ROOT CAUSE FOUND — nQ16 label-perspective inversion
- n16 "plays backwards" diagnosis chain (all evidence in DATA-CONVENTIONS.md):
  engine wiring CORRECT (source-verified EvalFile->slot map); eval signs CORRECT but
  magnitudes 4-7x compressed (queen-up = +246 vs nQ13 +1069, worst on black-to-move).
  Bin-sample audit: labels at 0.34x scale, sign chaos. Source audit: segments STM
  (31/36), puzzles STM (65/74), MINIATURES WHITE-POV (39/40, SF8-labeled, 43.5M rows),
  degm white-pov. assemble16r packed all verbatim into STM-pov bins => every
  black-to-move miniature row inverted. Val corr was self-consistently positive (trap).
- FIX: pov-aware emit in assemble16r.py + assemble16anti.py (white sources negated for
  black-stm at pack; NO relabeling needed). Committed + pushed.
- BINS REBUILDING: box /srv/workspace/chess-active/train16v2/ (assemble.log), fixed
  assembler from repo, launched ~01:5x. Re-run match16/audit_bin_labels.py on v2 bins
  BEFORE training (expect stm-sign ~100%, scale ~1.0).
- RETRAINING not launched (GPU capacity ruling pending): A) box-only sequential,
  6 miniature-flooded experts first (op_even_l0/l1/l2p, op_pawnimb, mg_safe/unsafe)
  ~2-3 days; B) wait for sparks => 16-parallel ~1 day. Anti16 nets equally invalid
  (same bug) — re-assemble anti after main rebuild.
- nQ recreation: YES, recipe-complete (DATA-CONVENTIONS.md) — all sources on cold
  RAID, pack_expert_bins.py + nnue-pytorch + nq13-engine in repo.
- evalsigntest.py = fast engine-level regression gate (known-sign positions) — run it
  after ANY net swap before matchmaking.

## 2026-10-07 LATER: PILOT VERDICT + SF17 RELABEL + UI FIXES
- UI: game-switch now scrolls movelist to top (first moves visible) + resets chart zoom +
  per-render try/catch. Verified in real browser. (index.html committed.)
- SF17 relabel of all 43.5M miniature labels RUNNING on box (relabel17b.py, 16 workers,
  stm-pov output, self-validating; ETA ~20h; out: miniature_labels17.tsv). First attempt
  (relabel17.py) had a worker-arg unpack bug and wrote garbage — caught by row-shape check,
  deleted, rewritten. mini_label.py used SF8 because it pointed at /usr/games/stockfish —
  the SF17 ruling was applied to farm labelers but missed this box-side script.
- PILOT (operator: verify before full training): 60k piece_down positions, SF17 d12 stm-pov,
  fixed emit, 35-min train on box. Verdict (pilot_verdict.py, net in piece_down slot):
  queen-up +809/-759 (broken v1: +246/-142; truth ~±900), rook-down -341/+466.
  PIPELINE FIXED — corrected data produces sane material evals from a tiny budget.
- train16v2 assembly COMPLETE (perspective-fixed, mini labels still SF8).
  v3 (SF17 labels everywhere) after relabel17 finishes: assemble with MINILBL=
  miniature_labels17.tsv, minis pov default stm. RETRAIN from v3 (or v2 interim) once
  GPU capacity ruled. audit gates now: source_audit (pre-pack law), audit_bin_labels
  (post-pack), evalsigntest/pilot_verdict (engine-level) — the routing-only audit of
  10-06 could not catch label perspective; these three each would have.
