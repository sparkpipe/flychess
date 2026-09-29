#!/bin/bash
# PROPER RETRAIN: bin-sized epochs (epoch-size), validation every epoch,
# train while val improves, generous ceilings. Equal-time law replaced by
# operator ruling "continue as long as it makes a better model".
set -u
R=/mnt/cold-raid6/chess-audit
TR=$R/train_val/train
VA=$R/train_val/val
RUNS=$R/runs2
NETS=$R/nets2
TP=/home/spec/nnue-pytorch
mkdir -p "$RUNS" "$NETS"
cd "$TP"

train_one() {
  local name=$1 valsize=$2 maxep=$3
  echo "=== TRAIN $name (epoch-sized, max $maxep epochs) $(date -u) ==="
  python3 train.py "$TR/$name.bin" \
    --validation-datasets "$VA/$name.bin" \
    --validation-size "$valsize" \
    --check-val-every-n-epoch 1 \
    --epoch-size "$(python3 -c "import os;print(max(1000, os.path.getsize('$TR/$name.bin')//40))")" \
    --batch-size 8192 --max-time 02:00:00 --max-epochs "$maxep" \
    --default-root-dir "$RUNS/$name" \
    >> "$RUNS/$name.trainlog" 2>&1
  local ck
  ck=$(ls -t "$RUNS/$name"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    python3 serialize.py "$ck" "$NETS/$name.nnue" >> "$RUNS/$name.trainlog" 2>&1 \
      && echo "SERIALIZED $name"
  else
    echo "NO CKPT $name"
  fi
}

train_one balanced_l0  50000 60
train_one balanced_l1  50000 60
train_one nvb          50000 60
train_one exchanges    50000 60
train_one balanced_l2  50000 80
train_one oppb         30000 80
train_one bvr          30000 120
train_one dvoretsky    30000 120
train_one balanced_l3  25000 120
train_one nvr          25000 120
train_one rv2m         16000 200
train_one qvmat         8000 300
train_one tactics     100000 30
train_one tb_training  50000 60

touch "$R/RETRAIN_DONE"
echo "=== RETRAIN COMPLETE $(date -u) ==="
