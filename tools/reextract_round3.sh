#!/bin/bash
# Re-extract the stack cache from ROUND-3 expert checkpoints after ROUND3_DONE.
R=/mnt/cold-raid6/chess-audit
while [ ! -f "$R/ROUND3_DONE" ]; do sleep 300; done
echo "=== re-extraction from round 3 $(date -u) ==="
rm -rf "$R/stack_cache_r3"
cd /home/spec/chess-lab
# point extraction at runs3 checkpoints (best-val)
for name in balanced_l0 balanced_l1 nvb exchanges balanced_l2 oppb bvr \
            dvoretsky balanced_l3 nvr rv2m qvmat; do
  best=$(python3 - "/mnt/cold-raid6/chess-audit/runs3/$name.trainlog" << 'PYEOF'
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
       awk -v b="$best" '$1<=b {c=$2} END{print c}')
  [ -z "$ck" ] && ck=$(ls $R/runs3/$name/lightning_logs/version_*/checkpoints/last.ckpt | head -1)
  echo "linking $name -> $(basename $ck)"
  mkdir -p $R/stack_cache_r3_ckpts
  ln -sf "$ck" $R/stack_cache_r3_ckpts/$name.ckpt
done
ln -sf $R/runs/tactics/lightning_logs/version_0/checkpoints/last.ckpt $R/stack_cache_r3_ckpts/tactics.ckpt

# now run extraction pointing at round-3 ckpts
sed "s|RUNS = \"/mnt/cold-raid6/chess-audit/runs\"|RUNS = \"/mnt/cold-raid6/chess-audit/stack_cache_r3_ckpts\"|; s|f\"{RUNS}/{name}/lightning_logs/version_\*/checkpoints/last.ckpt\"|f\"{RUNS}/{name}.ckpt\"|" \
  /home/spec/chess-lab/tools/extract_stack_features.py > /home/spec/chess-lab/tools/extract_r3.py
python3 tools/extract_r3.py \
  $R/expert_bins_both $R/stack_cache_r3 \
  >> $R/stack_cache_r3.log 2>&1
touch "$R/REEXTRACT_DONE"
echo "=== re-extraction complete $(date -u) ==="
