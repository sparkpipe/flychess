#!/bin/bash
# Serialize from BEST-val checkpoint (runs3) -> nets3.
R=/mnt/cold-raid6/chess-audit
name=$1
cd /home/spec/nnue-pytorch
best_ep=$(python3 - "$R/runs3/$name.trainlog" << 'PYEOF'
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
ck=$(ls $R/runs3/$name/lightning_logs/version_*/checkpoints/epoch=*-step=*.ckpt 2>/dev/null | \
     sed -E "s/.*epoch=([0-9]+)-.*/\1 &/" | sort -n | \
     awk -v b="$best_ep" '$1<=b {c=$2} END{print c}')
[ -z "$ck" ] && ck=$(ls $R/runs3/$name/lightning_logs/version_*/checkpoints/last.ckpt | head -1)
python3 serialize.py "$ck" "$R/nets3/$name.nnue" >> $R/bestpick3.log 2>&1 \
  && echo "OK3 $name (epoch $best_ep)" || echo "FAIL3 $name"
