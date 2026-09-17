# FLYCHESS HANDOFF — runtime state for the next session (2026-09-17 22:40 UTC)

*Read DESIGN.md first (same directory; also at ~/chess-lab/DESIGN.md on rtx5090) —
the architecture, laws, and roadmap. This doc = the RUNTIME state and the open threads.*

## Where the program is (one paragraph)

The fly plays chess through a gated curriculum on a frozen Drosophila
connectome (MaleCNS, 188,778 neurons). The movement foundation (stages 1–4:
movement, blocking, protection, king-safety/pins/forks/discovered) is being
RETRAINED from scratch under the normalized dynamics (ANORM=1) + the full
adaptive stack — L1–L5 + stages 2–3 PASSED at pace parity-or-better vs the
saturated lineage; stage 4's fork took its floor-accepted marginal pass
(0.979) and DISCOVERED (the historical wall) is re-grinding now. The
imagination layer (the world model) graduated twice (0.9977/0.9974 depth-1/2
verified). The eval layer (the universal differential + the danger veto) is
designed, implemented in the rig, and awaits stage 5's semantics. The H01
human cortical connectome (910G) is downloaded and format-decoded; the edge
table extraction is the next build. The prior-art target: mlabonne/chessfly
(30.4% SF agreement, MAE 0.081 — the adult FlyWire with trainable synapses).

## The running processes (rtx5090:~/chess-lab)

| process | state | notes |
|---|---|---|
| `curriculum_v3.sh` (ANORM=1) | RUNNING, pid ~1544918 | stage 4 re-grind: discovered at ~0.854 climbing; fork took marginal at cap (0.979); per-lesson checkpoints fly_cb_v3_s4.pt |
| fly_imagine retrain | RELAUNCHED 22:4x after the fi-eval_battery NameError fix | I1 regrind; ISTATE=fly_imagines_retrain.pt, ISKIP=1, ILR=1.5e-4; log imagine_retrain.out |
| H01 extraction | DOWN by design | the relationship-file decode is the blocker; see below |

## The v3+ANORM chain — what to watch

- The chain log: curriculum_v3.log; the per-stage outputs v3_s2/s3/s4.out.
- The discovered battery: if it gates (0.98) → stage 4 completes → the
  normalization A/B verdict = PASSED → the differential build (D1/D2) starts.
- If it exhausts below floor (best <0.9745 at 4000) → the chain FAILs → await
  adjustment (do not blind-relaunch; the tiered acceptance already ran).
- The exhaustion-acceptance is NOW in the S4 loop (the fork's 0.979 cap-exit
  was the trigger for adding it — the pre-patch code hard-exited).

## The imagination retrain — the open bug just fixed

The retrain crashed at its first I2 eval: `NameError: fi is not defined` —
my LP-wiring patch referenced `fi.eval_battery` inside fly_imagine.py
itself. Fixed to the module-local `eval_battery` and relaunched (resumes
from fly_imagines_retrain.pt = the I1-pass weights). If it crashes again at
an eval: check for OTHER stale self-references in the LP block.

## The H01 extraction — the state and the blocker

- The id enumeration WORKS: 190,576,758 synapse ids via the per-shard
  list_labels (the reader's list_labels(fn, path="") — the path arg must be
  EMPTY or the join doubles: by_id/by_id/00.shard).
- by_id/00.shard was a FAILED DOWNLOAD (0 bytes local vs 345,196,050 on GCS,
  md5 HF0WPqOL7jF+D0Wa7Iaopw==) — RE-FETCHED and verified. The receipt's
  aggregate-byte check masked it; the full per-file audit found only the
  intentionally-skipped 4nm_raw prefix missing.
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

1. **Stage 4 completion** (discovered) — the A/B verdict.
2. **The differential head D1/D2** (the 4096-slot option-space differential,
   rule-derived training) — attaches to the completed movement weights.
3. **Stage 5 mates** + the dense-board curriculum (the 0.963 OOD gap).
4. **The Lichess CC0 corpus** fetch (4.4M SF-annotated positions) — the D3
   calibration + chessfly-comparable training.
5. **The H01 edge-table extraction** + the book-memory experiment.
6. **The opening module** (the operator's gambit spec: BD/Von Popiel/
   Smith-Morra/Budapest/Englund/Latvian as the exemplars; the initiative
   ledger bands +1/+2/+3; the asymmetric scoring win 1.0/draw 0.4-0.45).
7. **Stage 6 Dvoretsky** (tablebase-gated, zugzwang-correct).

## The benchmark ladder

- Current: SF-agreement 2.5–5% (the untrained eval layer).
- chessfly's published: 30.4% SF-agreement, MAE 0.081 (the direct prior art).
- The mid-term target: ≥30.4% with our stricter substrate + the provable
  curriculum; then beyond with the imagination-for-depth + H01 search.

## The laws recap (the full list in DESIGN.md)

Scaffold-free gates; the spot-test asymmetry; the at-rail probe; stable
seeds; the tiered acceptance (0.98/0.975/0.95); the v3 chain never killed
mid-lesson; pkill alone; the local-canonical patch flow; the PR flow
(branch → test → PR #1 → merge at stability); fresh-tail liveness.

## The key artifacts

- The checkpoints: fly_cb_v3_l1..l6.pt, fly_cb_v3_s2/s3/s4.pt (the ANORM
  lineage); fly_imagines_retrain.pt (the imagination deliverable);
  fly_sees_current.pt (the vision deliverable); the graduated arms archived
  at /mnt/model-warm/flychess-archive/graduated-arms/; the saturated lineage
  at archive/v3_saturated/ on the node.
- The PR: sparkpipe/flychess#1 (adaptive-stack-2026-09 → master).
- The design: DESIGN.md (repo + node). The prior art: mlabonne/chessfly (HF).
- The H01: /mnt/model-warm/human-h01-connectome (910G, verified); the ids
  checkpointed at h01_ids.npy; the extraction h01_extract_full.py.
