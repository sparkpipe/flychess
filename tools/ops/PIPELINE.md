# Training pipeline — exact reproduction runbook

Reproduces the general3C curriculum training (and the general3 lineage it
stands on). Every stage names its inputs, commands, and recorded artifact
hashes. Paths are the current box layout (`/mnt/flychess/...`); scripts
written before the 2026-10-16 migration hardcode the equivalent
`/srv/workspace/...` paths — translate when rerunning.

## Environment (recorded 2026-10-16, rtx5090)

- Python 3.14.4 (system /usr/bin/python3.14)
- torch 2.11.0+cu128, numpy 2.5.2, tyro 1.0.16, chess 1.11.2,
  tqdm 4.70.0, torchmetrics 1.9.0, psutil 7.2.2
- GPU: NVIDIA GeForce RTX 5090 (CUDA 12.8)
- C++ data loader — build once per checkout:
  `cd tools/nnue-pytorch/data_loader/cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
  → `build/libtraining_data_loader.so` (never commit build/)

## Stage 1 — frozen inputs

| Artifact | Size (B) | sha256 |
|---|---|---|
| store/games.db (positions + sf17 labels d10–d25) | 117,358,514,176 | (117 GB — hash on demand) |
| clean_short_wins.json (3,916 games, from find_clean_wins_spec.py) | 805,116 | e879921b156e3f52dec3e67bf33637a5baf562a38747dc51c044f9f3a11144ec |
| gambit_aug_positions.txt (cold-raid6 archive) | 69,538,983 | 4b50ddfe223e7e12c1d7b472641ef6106a440eef277fa7c29f50ae1ec35efa44 |

Both builders also `sys.path.insert` the chess-lab tools for
`pack_expert_bins.pack_sfen/pack_move`.

## Stage 2 — general3 bins (single-depth parent set)

`tools/session/build_general3_v2.py` — OTB segments d20 + puzzles d20 +
clean-short-win positions + gambit winner positions duplicated.

| Artifact | Records | sha256 |
|---|---|---|
| bins-general3/general3.train.bin | 14,386,163 | 8857ae254c295b6bb44cf3e6984cec1a01239e02e2d46ae0321f230ae8c244b8 |
| bins-general3/general3.val.bin | 774,418 | 3e8a5aa97f4bbe4539c183282524490918a7708ff8806d899de0b1edcba8b578 |

Record = 40 B (bin format via pack_sfen). 575,446,520 / 30,976,720 B.

## Stage 3 — curriculum bins d10..d20

`tools/ops/build_curriculum_bins.py` — same positions and sources as
general3; only the label depth changes. d10/d11: 14,160,010 records
(566,400,400 B; some positions lack shallow labels); d12–d20: 14,386,163.

Sample hashes (train bins):
- d10: c5bfe8320f1d187907d0084ebd007de687a4b52766b36273ada1741167987885
- d15: 686a26806f6c102f59a6f6c12f68e75e7925909d2ad7be3ea9333bcd40de2e6f
- d20: c44457f72dabf8eece90d3c0eafd69de609804f8049404ab243ab9e9a0ee18ee

## Stage 4 — curriculum training

`tools/ops/train_general3c.sh` (fresh) / `tools/ops/resume_general3c.sh`
(continues at the highest stage holding a last.ckpt). Per stage:

```
python3 train.py bins-curriculum/dD/general3c.train.bin \
  --validation-datasets bins-curriculum/dD/general3c.val.bin \
  --validation-size 50000 --check-val-every-n-epoch 1 \
  --epoch-size <records> --batch-size 4096 \
  --max-time 00:06:00:00 --max-epochs 100000 \
  --network-save-period 1 --save_top_k 1 \
  [--resume-from-checkpoint <prev-stage best|own last.ckpt>] \
  --random-fen-skipping 3 --default-root-dir runs-general3c/stage_dD
```

Protocol: d10→d20 in order; each stage runs to plateau (15 stale
validations after the best val_corr) or the 6 h cap, whichever first; the
next stage warm-starts from this stage's best val_loss checkpoint. Seed 42
(torch/random/numpy all seeded). Measured ~14 min/epoch on the 5090.

Reference trajectory: d10 0.3974@e4 → d11 0.4413@e42 (capped) →
d12 0.4252@e50 → d13… (val_corr not comparable across depths).

## Stage 5 — serialize

`python3 tools/nnue-pytorch/serialize.py <best.ckpt> runs-general3c/best.nnue`
(n16-fork net format — required; vanilla SF17 terminates on these nets).

## Stage 6 — evaluation

Match scripts in tools/ops (match_general3*.sh): fastchess, 6-game
3open.epd opening set, color-reversed; 2 s+200 ms margin and fixed-d20
protocols. Reference anchors: SF8 3223, SF17 ≈ 3470.

## Reproducibility notes

- RNGs are seeded (42). GPU kernels are not bitwise deterministic, so
  re-training yields statistically equivalent, not byte-identical, nets.
- Data identity IS byte-exact: verify the bin sha256 table above before
  training; a mismatch means the data stage changed.
- Builder scripts predate the /mnt migration and hardcode /srv paths
  internally (DB, chess-lab import, gambit file) — portability fix is a
  follow-up.
- games.db is the product of the dbnet fleet sweeps (tools/dbnet, in repo)
  and the AJ labeling; treat the 117 GB file as the frozen ground truth.
