#!/bin/bash
# Extra 10-min round for lacking experts (low corr = big-bin undertraining), fresh dir.
cd /srv/workspace/flychess/src/nnue-pytorch
B=/srv/workspace/chess-active/bins-db-v1
R=/srv/workspace/chess-active/runs-extra10
mkdir -p $R
for e in piece_down nvb dv_Q dv_R mvr oppb tb; do
  mkdir -p $R/$e
  TR=$(( $(stat -c %s $B/$e.train.bin) / 40 ))
  VR=$(( $(stat -c %s $B/$e.val.bin) / 40 )); [ $VR -gt 30000 ] && VR=30000
  echo "[$(date +%H:%M)] $e extra start (epoch-size $TR)"
  python3 train.py $B/$e.train.bin --validation-datasets $B/$e.val.bin \
    --validation-size $VR --check-val-every-n-epoch 1 --epoch-size $TR \
    --batch-size 4096 --max-time 00:00:10:00 --max-epochs 100000 \
    --network-save-period 100000 --random-fen-skipping 3 \
    --default-root-dir $R/$e >> $R/$e/train.log 2>&1
  ck=$(ls -t $R/$e/lightning_logs/version_*/checkpoints/last.ckpt $R/$e/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
  [ -n "$ck" ] && python3 serialize.py "$ck" $R/$e/final.nnue >/dev/null 2>&1 && echo "$e extra done"
done
echo EXTRA-ROUND-DONE
