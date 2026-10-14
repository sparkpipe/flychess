#!/bin/bash
# general20: train from general1's best checkpoint. No monitor — Lightning
# keeps the best checkpoint by itself (save_top_k 1); plateau is judged
# offline from metrics.csv. 48h cap is the only stop.
cd /srv/workspace/flychess/src/nnue-pytorch
B=/srv/workspace/chess-active/bins-general20
R=/srv/workspace/chess-active/runs-general20
CKPT=/srv/workspace/chess-active/runs-general1/lightning_logs/version_0/checkpoints/epoch=46-step=45872.ckpt
mkdir -p $R
TR=$(( $(stat -c %s $B/general20.train.bin) / 40 ))
echo "[$(date +%H:%M)] general20: epoch-size $TR, warm from ${CKPT##*/}, no monitor"
python3 train.py $B/general20.train.bin --validation-datasets $B/general20.val.bin \
  --validation-size 50000 --check-val-every-n-epoch 1 --epoch-size $TR \
  --batch-size 4096 --max-time 00:48:00:00 --max-epochs 100000 \
  --network-save-period 1 --save_top_k 1 \
  --resume-from-checkpoint $CKPT \
  --random-fen-skipping 3 --default-root-dir $R >> $R/train.log 2>&1
echo "[$(date +%H:%M)] general20 trainer exited"
