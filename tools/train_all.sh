#!/bin/bash
# EQUAL-TIME- PER-EXPERT TRAINING (ruling 2026-09-28):
#   - 30 min wall-clock per expert (--max-time), small bins get more iterations
#   - 10GB VRAM cap (batch 8192; verify on first run)
#   - held-out 5% validation per bin; val loss logged for overtraining check
#   - serialize each to nets/<expert>.nnue
set -u
R=/mnt/cold-raid6/chess-audit
TR=$R/train_val/train
VA=$R/train_val/val
RUNS=$R/runs
NETS=$R/nets
TP=/home/spec/nnue-pytorch
PER=${PER:-00:00:30:00}      # equal time per expert
BS=${BS:-8192}               # ~5-8GB VRAM

mkdir -p "$RUNS" "$NETS"
cd "$TP"

train_one() {
  local name=$1 valsize=$2 epochcap=$3
  echo "=== TRAIN $name (val $valsize, max-epochs $epochcap) $(date -u) ==="
  python3 train.py "$TR/$name.bin" \
    --validation-datasets "$VA/$name.bin" \
    --validation-size "$valsize" \
    --check-val-every-n-epoch 5 \
    --batch-size "$BS" --max-time "$PER" --max-epochs "$epochcap" \
    --default-root-dir "$RUNS/$name" \
    > "$RUNS/$name.trainlog" 2>&1
  local ck
  ck=$(ls -t "$RUNS/$name"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "NO CKPT $name"; return 1; }
  python3 serialize.py "$ck" "$NETS/$name.nnue" \
    --features "HalfKAv2_hm^" >> "$RUNS/$name.trainlog" 2>&1 \
    && echo "SERIALIZED $name -> $NETS/$name.nnue"
}

# expert: val-size, epoch cap (small bins capped to limit overfit passes)
train_one balanced_l0  50000 400
train_one balanced_l1  50000 400
train_one nvb          50000 400
train_one exchanges    50000 400
train_one balanced_l2  50000 400
train_one oppb         50000 400
train_one bvr          30000 800
train_one dvoretsky    30000 800
train_one balanced_l3  25000 800
train_one nvr          25000 800
train_one rv2m         16000 1600
train_one qvmat         8000 3200
train_one tactics     100000 60
train_one tb           50000 200

touch "$R/TRAINING_DONE"
echo "=== ALL TRAINING COMPLETE $(date -u) ==="
