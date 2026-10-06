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

## Live training (continues)
- 15 sparks + box (op_even_l1). Run dirs: `~/run16_<expert>/` on each spark,
  `/srv/workspace/chess-active/run16_op_even_l1/` on box. skip=3, no time limit.
- Spark→expert map: spark0=tb 1=mvr 2=rv2m 3=qvmat 4=nvb 5=piece_down 6=oppb 7=dv_Q 8=dv_R
  9=dv_rest a=op_pawnimb b=op_even_l0 d=op_even_l2p e=mg_unsafe f=mg_safe, box=op_even_l1, sparkc=none.
- Watchers: `/tmp/watch_<e>.sh` per spark (per-epoch .nnue export+delete 1.9GB ckpt, hourly
  wallclock ckpt keep-3). Some restarted runs (tb,mvr,dv_Q,oppb,mg_safe,qvmat? watch state) may
  lack watchers — CHECK `pgrep -f watch_` per spark if ckpts pile up (disk-full killed 11 runs already).
- Keepalive cron on box every 30min: `/tmp/fleet_keepalive.sh` (prunes nets, backs up newest ckpt
  + nets to `/mnt/cold-raid6/chess-audit/train16_backup/<expert>/`).
- Trainer val metrics now report val_corr, val_mae, val_median in TRUE cp (fixed in
  `/srv/workspace/flychess/src/nnue-pytorch/model/nnue.py` — box copy only; spark copies have OLD
  double-scale val_mae, corr unaffected).

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
