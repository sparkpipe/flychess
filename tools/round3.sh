#!/bin/bash
# ROUND 3: merge each expert's OTC bin with its domain slice of the tactics
# data (sharp game), rebuild val sets proportionally, retrain all 12.
# Tactics + TB experts unchanged (already trained on their full data).
set -u
R=/mnt/cold-raid6/chess-audit
TP=/home/spec/nnue-pytorch
RUNS=$R/runs3
NETS=$R/nets3
MER=$R/round3_train
mkdir -p "$RUNS" "$NETS" "$MER"
cd "$TP"

build_merged() {
  local name=$1 dom=$2
  mkdir -p $MER/$name
  cat $R/train_val/train/$name.bin $R/tactics_split/$dom.bin > $MER/$name/train.bin
  # val: existing OTB val + 5% of the tactics slice
  python3 - "$R" "$MER" "$name" "$dom" << 'PYEOF'
import os, sys
R, MER, name, dom = sys.argv[1:5]
src = f"{R}/tactics_split/{dom}.bin"
out = f"{MER}/{name}/val_tactics.bin"
n = os.path.getsize(src) // 40
with open(src, "rb") as f, open(out, "wb") as w:
    for i in range(n):
        r = f.read(40)
        if i % 20 == 0:
            w.write(r)
PYEOF
  cat $R/train_val/val/$name.bin $MER/$name/val_tactics.bin > $MER/$name/val.bin
  echo "merged $name: $(wc -c < $MER/$name/train.bin) train bytes, $(wc -c < $MER/$name/val.bin) val bytes"
}

train_one() {
  local name=$1 valsize=$2 maxep=$3
  echo "=== TRAIN3 $name (max $maxep epochs) $(date -u) ==="
  python3 train.py $MER/$name/train.bin \
    --validation-datasets $MER/$name/val.bin \
    --validation-size "$valsize" \
    --check-val-every-n-epoch 1 \
    --epoch-size "$(python3 -c "import os;print(max(1000, os.path.getsize('$MER/$name/train.bin')//40))")" \
    --batch-size 8192 --max-time 00:02:00:00 --max-epochs "$maxep" \
    --default-root-dir "$RUNS/$name" \
    >> "$RUNS/$name.trainlog" 2>&1
  bash /home/spec/chess-lab/tools/best_pick3.sh "$name"
}

build_merged balanced_l0 balanced_l0
build_merged balanced_l1 balanced_l1
build_merged nvb nvb
build_merged exchanges balanced_l0   # transient class: use l0 slice + own OTB
build_merged balanced_l2 balanced_l2
build_merged oppb oppb
build_merged bvr bvr
build_merged dvoretsky dvoretsky
build_merged balanced_l3 balanced_l3
build_merged nvr nvr
build_merged rv2m rv2m
build_merged qvmat qvmat

train_one balanced_l0  50000 25
train_one balanced_l1  50000 30
train_one nvb          50000 40
train_one exchanges    50000 30
train_one balanced_l2  50000 40
train_one oppb         30000 60
train_one bvr          30000 100
train_one dvoretsky    30000 40
train_one balanced_l3  25000 150
train_one nvr          25000 100
train_one rv2m         16000 150
train_one qvmat         8000 250

cp $R/nets2/tactics.nnue $NETS/tactics.nnue 2>/dev/null || \
  cp $R/nets/tactics.nnue $NETS/tactics.nnue
cp $R/nets2/tb_training.nnue $NETS/tb_training.nnue 2>/dev/null || true
touch "$R/ROUND3_DONE"
echo "=== ROUND 3 COMPLETE $(date -u) ==="
