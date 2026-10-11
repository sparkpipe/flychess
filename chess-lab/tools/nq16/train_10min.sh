#!/bin/bash
# 10-min training per expert from bins-db-v1 (operator order), one final .nnue each.
cd /srv/workspace/flychess/src/nnue-pytorch
B=/srv/workspace/chess-active/bins-db-v1
R=/srv/workspace/chess-active/runs-10min
mkdir -p $R
for e in tb mvr rv2m qvmat nvb piece_down oppb dv_Q dv_R dv_rest op_pawnimb op_even_l0 op_even_l1 op_even_l2p mg_unsafe mg_safe; do
  [ -f $B/$e.train.bin ] || { echo "$e MISSING-BIN"; continue; }
  [ -f $R/$e/final.nnue ] && { echo "$e already done"; continue; }
  mkdir -p $R/$e
  TR=$(( $(stat -c %s $B/$e.train.bin) / 40 ))
  VR=$(( $(stat -c %s $B/$e.val.bin) / 40 )); [ $VR -gt 30000 ] && VR=30000
  echo "[$(date +%H:%M)] $e start (epoch-size $TR, val $VR)"
  python3 train.py $B/$e.train.bin --validation-datasets $B/$e.val.bin \
    --validation-size $VR --check-val-every-n-epoch 1 --epoch-size $TR \
    --batch-size 4096 --max-time 00:00:10:00 --max-epochs 100000 \
    --network-save-period 100000 --random-fen-skipping 3 \
    --default-root-dir $R/$e >> $R/$e/train.log 2>&1
  ck=$(ls -t $R/$e/lightning_logs/version_*/checkpoints/last.ckpt $R/$e/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    python3 serialize.py "$ck" $R/$e/final.nnue >/dev/null 2>&1 && echo "$e done -> final.nnue"
  else
    echo "$e NO-CHECKPOINT"
  fi
done
echo ALL-16-DONE
