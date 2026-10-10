#!/bin/bash
# Serialize each finished expert from its BEST-validation checkpoint
# (nearest saved checkpoint at/before the min-val epoch), not last.ckpt.
R=/mnt/cold-raid6/chess-audit
cd /home/spec/nnue-pytorch
for name in "$@"; do
  log=$R/runs2/$name.trainlog
  best_ep=$(python3 - "$log" << 'PYEOF'
import re, sys
best, bep = 1e9, 0
for line in open(sys.argv[1]):
    m = re.match(r"Epoch (\d+) \(Val\): \[val_loss_epoch=([0-9.]+)\]", line)
    if m:
        ep, v = int(m.group(1)), float(m.group(2))
        if v < best:
            best, bep = v, ep
print(bep)
PYEOF
)
  ck=$(ls $R/runs2/$name/lightning_logs/version_*/checkpoints/epoch=*-step=*.ckpt 2>/dev/null | \
       sed -E "s/.*epoch=([0-9]+)-.*/\1 &/" | sort -n | \
       awk -v b="$best_ep" '$1<=b {c=$2} END{print c}')
  [ -z "$ck" ] && ck=$(ls $R/runs2/$name/lightning_logs/version_*/checkpoints/last.ckpt | head -1)
  python3 serialize.py "$ck" "$R/nets2/$name.nnue" >> $R/bestpick.log 2>&1 \
    && echo "OK $name (best epoch $best_ep)" || echo "FAIL $name"
done
