# FLYCHESS HANDOFF — runtime state for the next session (2026-09-17 23:05 UTC)

*Read DESIGN.md first (same directory; also at ~/chess-lab/DESIGN.md on rtx5090) —
the architecture, laws, and roadmap. This doc = the RUNTIME state and the open threads.*

## Where the program is (one paragraph)

The fly plays chess through a gated curriculum on a frozen Drosophila
connectome (MaleCNS, 188,778 neurons). **THE MOVEMENT FOUNDATION IS
COMPLETE: the v3+ANORM chain finished 2026-09-17 22:59 UTC — all 6 lessons
+ stages 2/3/4 PASSED, final sweep 27/27 batteries pass (worst 0.982
s1:castle; fork 1.0, discovered 1.0, pin 1.0). PR #1 MERGED to master.**
The last wall (discovered at 0.958 for 5.5h/193 plateau cycles) was a
BATTERY ANSWER-KEY BUG, not capacity: the generator's discovery test used
the slider's FULL attack mask (rejecting chess-valid discoveries that land
on the unblocked perpendicular ray), and separately rejected
capture-of-target — but the model correctly prefers winning the queen
outright. Both fixed (segment test + capture acceptance), verified by
answer-key spot test, and the resumed weights passed everything at step 1.
The imagination layer graduated twice (0.9977/0.9974 depth-1/2 verified;
0.963 dense-OOD gap). The H01 human cortical connectome (910G) is
downloaded, audited (only the raw-EM layer missing — not needed), ids
enumerated; the edge table is the next build. Prior-art target:
mlabonne/chessfly (30.4% SF agreement, MAE 0.081).

## The running processes (rtx5090:~/chess-lab)

| process | state | notes |
|---|---|---|
| `curriculum_v3.sh` (ANORM=1) | **COMPLETE — do not relaunch** | log ends "CURRICULUM V3 COMPLETE"; checkpoints fly_cb_v3_l1..l6.pt + s2/s3/s4.pt |
| imagination retrain | GRADUATED (standing) | fly_imagines_retrain.pt; rig verdict loop_rig_verdict.json |
| H01 extraction | DOWN by design | the relationship-file decode is the blocker; see below |

## The discovered-battery lesson (the new laws)

1. **A concept battery must not penalize the better chess move.** When a
   discovery position also offers an outright win of the target piece, the
   model that takes the queen is RIGHT; the eval must accept both the
   discovery and the capture-of-target. (This was the second half of the
   0.958 pin — introduced by the first fix, caught live in the relaunch's
   tail, fixed the same hour.)
2. **Discovery = leaving the blocked slider→target segment**
   (`chess.between(slider,tgt) | {tgt}`), never "off the slider's full
   attack mask" — perpendicular-ray landings (e6f8/d4e6 class) are true
   discoveries. For a knight front EVERY move discovers (28/28 boards, 0
   excluded moves); the concept only bites for rook/bishop fronts
   (along-line stays stay rejected — 64 exclusions in the battery).
3. **The spot-test that matters for a battery bug is the ANSWER KEY
   itself** — regenerate the seeded battery, assert the failing picks are
   accepted and the concept guards still reject. Model-free, deterministic,
   one ssh. (The relaunch then passed at step 1, as predicted: 0.958 + 4
   flips + 6 capture-accepts → 1.0.)

## The H01 extraction — the state and the blocker

- The id enumeration WORKS: 190,576,758 synapse ids via the per-shard
  list_labels (the reader's list_labels(fn, path="") — the path arg must be
  EMPTY or the join doubles: by_id/by_id/00.shard).
- by_id/00.shard was a FAILED DOWNLOAD (0 bytes local vs 345,196,050 on GCS,
  md5 HF0WPqOL7jF+D0Wa7Iaopw==) — RE-FETCHED and verified. The full per-file
  audit: only 12 files MISSING, ALL in 4nm_raw (raw EM imagery, ~72G of
  ~1.1T) — NOT needed for the memory/search experiments; the dataset is
  complete for our purposes.
- **The blocker**: the pre/post_synaptic_cell relationship lookups return
  None for every synapse (edges=0 after 1.9M ids). The relationship files'
  internal organization does NOT match the (id>>10)&8191 minishard formula —
  the minishard-25 blob of 0.shard holds annotation ids 3.05e9–1.04e11 whose
  own minishard ≠ 25. The neuroglancer sharded.md spec was fetched and the
  layout is UNDERSTOOD (the shard index at the file HEAD: 2^minishard_bits
  (start,end) pairs; the minishard-index blobs gzipped; the 3-col cumsum
  decode with offsets +index_length) — but the relationship files' observed
  content contradicts the naive assignment. The next session: implement the
  reader per the spec + verify empirically against cloud-volume's own
  get_by_relationship on a KNOWN-good id, or adapt the chessfly-style
  extraction.
- The extraction script: h01_extract_full.py (v3 — the verified enumerate +
  batched get_by_id); the ids checkpointed at h01_ids.npy (190.6M).
- cloud-volume 12.14.4 in ~/chess-lab/.venv312 (py3.12 via uv; the system
  python 3.14 has no wheels). The info manifests were served gzipped —
  gunzipped in place. The np.hstack(a,b) cloud-volume bug shimmed.

## The queued builds (in order)

1. **Stage 5 mates** — imagination-for-depth + check semantics + the
   dense-board curriculum (the 0.963 OOD gap). The movement weights
   (fly_cb_v3_s4.pt) are the substrate.
2. **The differential head D1/D2** (the 4096-slot option-space differential,
   rule-derived training) — attaches to the completed movement weights.
3. **The Lichess CC0 corpus** fetch (4.4M SF-annotated positions) — the D3
   calibration + chessfly-comparable training.
4. **The H01 edge-table extraction** + the book-memory experiment.
5. **The opening module** (the operator's gambit spec: BD/Von Popiel/
   Smith-Morra/Budapest/Englund/Latvian as the exemplars; the initiative
   ledger bands +1/+2/+3; the asymmetric scoring win 1.0/draw 0.4-0.45).
6. **Stage 6 Dvoretsky** (tablebase-gated, zugzwang-correct).
7. **The personality adapters** (Morphy/Capablanca/Carlsen) + the H01
   search substrate.

## The benchmark ladder

- Current: SF-agreement 2.5–5% (the untrained eval layer).
- chessfly's published: 30.4% SF-agreement, MAE 0.081 (the direct prior art).
- The mid-term target: ≥30.4% with our stricter substrate + the provable
  curriculum; then beyond with the imagination-for-depth + H01 search.

## The laws recap (the full list in DESIGN.md)

Scaffold-free gates; the spot-test asymmetry; the at-rail probe; stable
seeds; the tiered acceptance (0.98/0.975/0.95); the v3 chain never killed
mid-lesson; pkill alone (bracket pattern — the bare pattern self-matches
the ssh-spawned shell, exit 255); the local-canonical patch flow; the PR
flow (branch → test → PR → merge at stability — MERGED 0b535eb, master is
canonical); fresh-tail liveness; gh merges need the ~/sparkpipe/.env PAT
(keychain creds are stale; the PAT also lacks the PR-merge API scope —
merge via git locally + push, the PR auto-closes).

## The key artifacts

- The checkpoints: fly_cb_v3_l1..l6.pt, fly_cb_v3_s2/s3/s4.pt (the ANORM
  lineage — THE movement deliverable); fly_imagines_retrain.pt (the
  imagination deliverable); fly_sees_current.pt (the vision deliverable);
  the graduated arms archived at /mnt/model-warm/flychess-archive/; the
  saturated lineage at archive/v3_saturated/ on the node.
- The repo: sparkpipe/flychess, master @ 0b535eb (PR #1 MERGED).
- The design: DESIGN.md (repo + node). The prior art: mlabonne/chessfly (HF).
- The H01: /mnt/model-warm/human-h01-connectome (910G, verified); the ids
  checkpointed at h01_ids.npy; the extraction h01_extract_full.py.

## The 8-hour unmonitored window (2026-09-18, operator: "stop for the day")

The driver automation is DELETED (operator ruling — training runs
unmonitored ~8h). State at window start:
- **Stage 5**: mate1 gate PASSED (1.00 on 200 held, all-mates key); the
  chain is in `S5 SWEEP CYCLE 1` under the operator's cycle protocol
  (repairs interleave 20% mate1 turns ONLY while held<0.98 → hold new set
  ≥0.98 → re-table → up to 3 cycles → `S5 NO-CONVERGENCE` + analysis dump
  and STOP). Check v3_s5.out first thing: STAGE 5 ALL MILESTONES PASSED /
  NO-CONVERGENCE / still mid-cycle. The branch stage5-mates merges to
  master via git when the PASS lands.
- **H01 extraction**: Ceph object fault FIXED (operator/sysadmin cleared
  by_id/01.shard — verified readable); extraction relaunched and healthy.
  NOTE: ~/.cloudfiles/locks accumulates stale lock files over long runs
  (4.1M files once) — clear it after the extraction completes. The edge
  table (relationship decode) remains the queued build after the raw
  extraction.
- **Training audit** (operator-directed, done 2026-09-18): stages 1-3
  labels rule-derived from chess.legal_moves (clean); stage 4 concept
  labels rule-verified at generation (kingless boards — SF inapplicable);
  stage 5 200/200 SF-validated (label is mate, SF agrees mate-in-1, SF
  best is a mate). The open training-side improvement: CE targets one
  labeled mate where several are correct — multi-target CE is the next
  fix when the SF-scored harness lands with D1/D2.
- **Disk**: root LV extended 100G→216G (128G unallocated absorbed; 36%
  used). The drafters LV (688G, /srv/drafters, retired dflash artifacts
  27G) awaits the operator's reclaim ruling.

## STAGE 6 PASSED — the v3 curriculum is COMPLETE (2026-09-18)

`STAGE 6 PASSED — tablebase endings internalized`: gate 0.99 at step 5500
(floor 0.98) after the reinforce-best fix took it off the 0.47 pin
(0.5875@500 → 0.8625@4000 → 0.99@5500). Graduated checkpoint archived at
/mnt/model-warm/flychess-archive/graduated-arms/fly_cb_v3_s6_graduated.pt;
live fly_cb_v3_s6.pt on the node. THE FULL LADDER: 6 lessons + stages
2/3/4 + stage 5 (KQvK mate-in-1, all-mates key) + stage 6 (tablebase
endings, graded state-change targets, zugzwang-correct). The next
builds, in order: the mates ladder via imagination depth (KRK technique,
mate-in-2, KBNvK W-maneuver), the differential head D1/D2 (+ the
SF-scored all-moves harness and multi-target CE), the Lichess CC0 fetch,
the opening module (gambit spec, asymmetric scoring), H01 experiments
after the extraction completes (relaunch recipe proven: tmpfs locks +
janitor; resume idempotent from 46 parts).
