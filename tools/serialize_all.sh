#!/bin/bash
# Serialize all expert nets from their checkpoints (correct lightning path).
R=/mnt/cold-raid6/chess-audit
cd /home/spec/nnue-pytorch
mkdir -p "$R/nets"
for name in balanced_l0 balanced_l1 nvb exchanges balanced_l2 oppb bvr \
            dvoretsky balanced_l3 nvr rv2m qvmat tactics tb_training; do
  ck=$(ls -t "$R/runs/$name"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    python3 serialize.py "$ck" "$R/nets/$name.nnue" --features "HalfKAv2_hm^" \
      >> "$R/serialize.log" 2>&1 \
      && echo "OK $name" || echo "FAIL $name"
  else
    echo "MISSING $name"
  fi
done
echo SERIALIZE_ALL_DONE
