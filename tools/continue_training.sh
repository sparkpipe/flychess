#!/bin/bash
# CONTINUATION: after ROUND3_DONE, keep resuming each expert from its
# checkpoint while validation still improves; stop on plateau; serialize
# best-val. Implements "train as long as it makes a better model".
set -u
R=/mnt/cold-raid6/chess-audit
TP=/home/spec/nnue-pytorch
cd "$TP"

still_improving() {
  # last-3 val min < prior-10 val min - 1% margin
  python3 - "$1" << 'PYEOF'
import re, sys
vals = []
for line in open(sys.argv[1]):
    m = re.match(r"Epoch (\d+) \(Val\): \[val_loss_epoch=([0-9.]+)\]", line)
    if m:
        vals.append(float(m.group(2)))
if len(vals) < 8:
    print("yes" if vals and vals[-1] <= min(vals) + 1e-6 else "no")
else:
    recent, prior = vals[-3:], vals[-13:-3]
    print("yes" if min(recent) < min(prior) * 0.99 else "no")
PYEOF
}

total_epochs() {
  grep -c "(Val)" "$1" 2>/dev/null || echo 0
}

while [ ! -f "$R/ROUND3_DONE" ]; do sleep 300; done
echo "=== continuation pass armed $(date -u) ==="

for name in balanced_l0 balanced_l1 nvb exchanges balanced_l2 oppb bvr \
            dvoretsky balanced_l3 nvr rv2m qvmat; do
  log=$R/runs3/$name.trainlog
  ck=$R/runs3/$name/lightning_logs/version_0/checkpoints/last.ckpt
  [ -f "$ck" ] || continue
  rounds=0
  while [ "$(still_improving "$log")" = "yes" ] && [ $rounds -lt 8 ]; do
    rounds=$((rounds+1))
    cur=$(total_epochs "$log")
    inc=15
    echo "=== CONT $name round $rounds (to epoch $((cur+inc))) $(date -u) ==="
    python3 train.py $R/round3_train/$name/train.bin \
      --validation-datasets $R/round3_train/$name/val.bin \
      --validation-size 30000 \
      --check-val-every-n-epoch 1 \
      --epoch-size "$(python3 -c "import os;print(max(1000, os.path.getsize('$R/round3_train/$name/train.bin')//40))")" \
      --batch-size 8192 --max-time 00:01:00:00 \
      --max-epochs $((cur+inc)) \
      --resume-from-checkpoint "$ck" \
      --default-root-dir $R/runs3/$name \
      >> "$log" 2>&1
    ck=$(ls -t $R/runs3/$name/lightning_logs/version_*/checkpoints/last.ckpt | head -1)
  done
  bash /home/spec/chess-lab/tools/best_pick3.sh "$name"
  echo "=== CONT $name done after $rounds continuations ==="
done
touch "$R/CONTINUATION_DONE"
echo "=== CONTINUATION COMPLETE $(date -u) ==="
