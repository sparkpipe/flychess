# FLYCHESS HANDOFF — complete state document

Last updated: 2026-09-16. Everything needed to continue without re-deriving.
Read this top to bottom before touching anything.

## 1. PROJECT

Train the real Drosophila connectome (MaleCNS, 188,778 neurons / 26M synapses,
`brain_graph.npz`) to play chess. Architecture: color-blind chess features →
sensory injection (26,933 neurons) → 2-step propagation → 4096-slot move
readout (`slot = from*64+to`) → legality BCE + move CE + 3-class pheromone
(lost/playable/won). Search when playing: FlyLeela PUCT MCTS (fly_mcts.py).

Operator's training philosophy (verbatim rulings — follow, don't reinterpret):
- "we do training in a sequence... iteratively from the ground up"
- Stage gates at 100% (or spec'd bar); nothing advances below gate
- "on a failure, STOP, then adjust what we do until that failure passes"
- "each training item is a milestone" — eval per training item, not in blocks
- "if each time you train from scratch it creates regressions, then you need
  to use a different part of the brain, or add another view of the same data"
- "make all training color blind" (mirror canonicalization; halves data)
- "we can always promote to queen" (underpromotions out of scope)
- "we MUST support castling and en passant"
- "the impulse should be to push. later layers will suppress it when
  appropriate" (promotion; attacked squares are distractors at instinct level)
- "for the soup cooking we need to feed in random order from all the corpus"
  (randomized combined corpus, NOT sequential per-piece blocks)
- "dont wait until all are done. stop at the first FAIL and do the usual
  corrections" (streaming FAIL-hunt in combined phase)
- "I want 45% win, 50% loss, 5% draws, not 90% draw" (draw contempt −0.25)
- Elo measurement ONLY after the tournament checklist passes
- "DO NOT CHANGE WHAT I ASK... certainly do not change without telling me"
- Launch-verify every trainer (first milestone lines within ~60s)

## 2. CURRENT STATE (rtx5090:~/chess-lab)

### Stage ladder status
- Stage 1 (piece movement, open board, no kings): ADAPTIVE PASSED all 6
  pieces + king added later; FROM-SCRATCH VERIFIED (verify1: VERIFIED).
  Backup: fly_cb_stage1_verified.pt (canonical pre-stage-2 base).
- Stage 2 (two pieces: blocking/capture, no kings): ADAPTIVE PASSED,
  FROM-SCRATCH VERIFIED (verify2: VERIFIED). Backup:
  fly_cb_stage2_verified.pt.
- Stage 3 (three pieces: protection): ADAPTIVE PASSED, FROM-SCRATCH
  VERIFIED (verify3: VERIFIED). Adaptive backup: fly_cb_stage3_adaptive.pt.
- Stage 4 (king safety/pins/forks/discovered): their_king/pin PASS step 1;
  fork passed after threat-count view fix; discovered EXHAUSTED at 0.958 on
  the OLD dense wiring. KNOWN ISSUE: stage-4 rebuild on retinotopic wiring
  is queued — see §5 next actions.
- Combined corpus (9 arms soup): crashed at first eval on eval_all signature
  bug (FIXED in code 5809874); needs relaunch — see §5.
- promo/castle single arms: castle PASSED (0.979 grind cleared);
  promo never passed its gate (stuck 0.952, then eval-mismatch found and
  fixed — needs rerun under fixed eval; its state file fly_promo.pt is from
  the stuck run).

### Working checkpoints (rtx5090:~/chess-lab/)
- fly_cb_stage1_verified.pt   stage-1 canonical base
- fly_cb_stage2_verified.pt   stage-2 canonical base
- fly_cb_stage3_adaptive.pt   stage-3 adaptive weights
- fly_souped.pt               soup of 8 arm-specialists (pre-combined-crash)
- fly_cb_m1.pt                working file of whatever arm ran last
- fly_cb_stage3_verified.pt   does NOT exist yet — stage-3 verify3 said
                              VERIFIED but stamping never ran (crash race);
                              re-stamp by copying the verified stage-3
                              weights before stage 4

### 259K exact-label tablebase pools
~/chess-lab/tbpools/*.jsonl — KPvK, KQvK, KRvK, KPvKP, KRvKP, KRPvKR,
KBvKP, KRvKB, KRvKN (built from lichess Syzygy API; entries carry fen,
best move, category, and full per-move child categories+DTZ for graded
targets). Pool builder: tb_pool.py (local-syzygy version would be faster;
API works but throttled).

## 3. THE STAGED CURRICULUM (fly_curriculum.py)

Stage generators (procedural, seeded, no data-quality doubts):
- S1: one piece, open board, NO KINGS (kingless legality verified).
  Operator spec verbatim. Pawns ranks 2-7.
- S2: two pieces: friendly blocker vs enemy piece (captures).
- S3: three pieces: protection — enemy target defended 50% of the time;
  CE target = capture when undefended, safe move when defended.
  ENEMY KINGS EXCLUDED (king-adjacency legality is S4 material — was
  capping S3 king at 0.984).
- S4: four concept modes: their_king (enemy-king legality, enemy to move),
  pin (pinned-piece legality), fork (knight double-attack CE),
  discovered (CE with target_set eval — accepts ANY valid discovery).
- Combined: randomized corpus across all 9 arms (6 stage-1 pieces + ep +
  promo + castle), universal slot-legality BCE + move CE + pheromone.

Gates: pairwise legality >= 0.99 (legality modes), CE argmax == 1.00
(concept modes), stable across two consecutive evals for combined.
Regression sweeps after EVERY pass: all prior stages × all pieces,
3 corrective rounds (300 steps each) then SWEEP-BLOCKED hard stop.

## 4. BUGS FOUND AND FIXED (do not re-introduce)

TRAINER:
- eval gate reset bug: gating evals on `nloss == 1` when nloss resets per
  print → eval every batch → 35 ips. Fixed with nstep counter.
- Memory: mp imap retained all results → 60GB RSS → kernel OOM killed the
  parent twice. Fixed: bounded apply_async submission, maxtasksperchild,
  fork context, del df after extraction.
- W_CLS negative weight: `mv_term * (1 - W_CLS)` with W_CLS=1.5 gave
  move-CE weight −0.5 → active unlearning (loss −45). Fixed: mv_term + cls.
- push_uci(Move) crashes in this python-chess (len() on Move) — always
  board.push(Move).
- mfb buffer was 8-wide; MOVE_DIMS now 17 (displacement features).
- numpy str in csv: pandas fine; crc32 holdout deterministic.
ENGINE (fly_mcts.py):
- Phase-2 terminal blindness: mating child evaluated with empty menu →
  scored 0 → mate invisible. Fixed: mate_v = +1 path.
- Backup double-count: leaf value added twice with sign flip → all
  terminal values cancelled to 0. Fixed: single-count backup.
- Root visits grew 1/batch → fixed to 1/sim.
- PUCT missing negation (child value must flip for parent perspective).
- eval_all() missing model arg (combined crash).
- eval_piece/eval_ce feature mismatch: promo push scored with zero mf
  (non-promotion move not pseudo-legal) → fixed with queen-suffix move.
- Promo generator infinite loop: Move(e7,e8) without promotion suffix is
  never legal → generator retried forever at 100% CPU. Fixed with
  promotion=QUEEN.
- Gen reject loop: is_game_over() true for K+B/K+N vs K boards → stage-1
  bishop/knight generation hung forever (2M attempts, 0 accepted). Fixed:
  insufficient-material boards are valid training boards.
- Threat-count color inversion: after push, mover's enemies are
  occupied_co[b.turn] (was inverted → pin degraded). Fixed.
- net patch region too small / wrong cells → rebuilt on disjoint hex2 band.
INFRA:
- fastchess: spin options REQUIRE min/max; float options must be type
  string; -repeat 0 → SIGFPE; never pipe fastchess through tail in a
  detached script (SIGPIPE kills it silently).
- pkill -f self-match kills your own ssh when the command line contains
  the target string elsewhere. Use bracket patterns AND separate commands.
- explorer.lichess.ovh = 401 globally; rybkachess dead; rebel13 broken
  cert; archive.org intermittently down. Tablebase API
  tablebase.lichess.ovh/standard?fen=... WORKS.

## 5. NEXT ACTIONS (in order)

1. RELAUNCH combined corpus (crash fixed):
   cd ~/chess-lab && STATE=/home/spec/chess-lab/fly_souped.pt RETINO=geo
   PIECES=king,rook,bishop,knight,queen,pawn,ep,promo,castle nohup
   python3 fly_curriculum.py 1 --steps 4000 > combined.out 2>&1 &
   Watch: first combined eval line within ~2 min (launch-verify).
2. When combined clears (all 9 batteries >= 0.99 stable): stage-4 rebuild
   on the retinotopic mainline (RETINO=geo), then stage-5 basic mates
   (mating_anchors already in trainer + KQvK/KRvK tablebase pool).
3. Stage 6: Dvoretsky chapter-by-chapter from tbpools (graded move-vector
   targets per operator spec: best=1.0, wasted tempo graded, dtz>=100
   cliff = draw-level, lost-win = −0.6, longer resistance in lost = less bad).
4. After ladder of stages: tournament checklist (checklist.py) with SF
   resistance at 100% bars — ONLY then Elo ladder + Swarm A/B.
5. Stage-3 verified stamp: verify3 printed VERIFIED but the stamp copy to
   fly_cb_stage3_verified.pt never executed (crash race). Re-stamp from
   the verified stage-3 weights before stage 4 launch.

## 6. KEY FILES (rtx5090:~/chess-lab/ and repo sparkpipe/flychess master)

- fly_curriculum.py   staged trainer (stages 1-4 + combined); INJECT env
                      swaps injection site; RETINO env for hex overlay;
                      WIRING=structured for family->site map; COMBINED=1
                      for randomized corpus mode; PIECES env filters arms.
- flyfeat_cb.py       color-blind features v2 (2746 dims); IMAGE=1 adds
                      768 raw bitmap dims (parked — see EXPERIMENTS.md).
- flyfeat_end.py      endgame concept extractor (55 dims, Dvoretsky spans).
- fly_mcts.py         FlyLeela engine (PUCT, menu-comparison value, trap
                      layer, stance controller, swarm mode).
- checklist.py        tournament gate: SF resistance, 100% bars.
- soup_merge.py       parameter average of arm checkpoints.
- tb_pool.py          exact-label pool builder (API or local syzygy).
- wiring_engineer.py  injection-site measurement (routing/capacity).
- EXPERIMENTS.md      experiment matrix + findings log.

## 7. MEASURED FINDINGS (the mapping-theory evidence so far)

- Retinotopic hex overlay: rook 948 -> 87 steps to gate (10x). Geo vs
  shuffled IDENTICAL at stage 1 -> within-site geometry neutral there;
  geometry test moves to stage 2/3 (neighbor relations).
- Wiring capacity 1.0 at every site (multiplexing not the limiter);
  sensory_periph is the WORST hub (50% self-locked) — site choice matters.
- Pseudo-legality binding view dissolved the rook S3 plateau (0.911 ->
  0.996; queen/pawn passed at step 1) — from->to relation was the missing
  view. Confirms operator's mapping-limit theory.
- King-adjacency geometry made king learnable from scratch (was stuck at
  0.08 forever without it).
- Transfer is additive: queen learned fastest (194 steps) after rook+bishop;
  knight (orthogonal leaper) independent; king needed its own adjacency.
- Anatomical decodability probe (pre-fix): NO region decoded legality under
  the old injection — the signal never entered the circuit. After fixes:
  stage batteries pass; region-level probes rerun pending.

## 8. OPERATOR CONTACT POINTS

- Repo: sparkpipe/flychess (FLYCHESS_PAT in ~/.env works for push).
- Monitor automation: automation-7fb7544d (10 min, stall+crash+completion
  on combined.out). Older automation-a6b0be89 deleted.
- sparks 3-8: chess-lab deployed, torch+deps installed (spark5 needed
  --force-reinstall cu128 wheel; was CPU-only build).
