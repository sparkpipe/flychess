#!/bin/bash
# Round 4: train the 3 largest experts on the 5090 GPU (balanced load
# with the 9 smaller experts on sparks). Train to plateau, best-val pick.
R=/mnt/cold-raid6/chess-audit
mkdir -p $R/runs4 $R/nets4
cd /srv/workspace/flychess/src/nnue-pytorch

for E in balanced_l0 balanced_l1 nvb; do
  BIN=$R/round3_train/$E/train.bin
  N=$(($(stat -c%s "$BIN") / 40))
  # epoch-size capped at 150K so epochs are consistent duration
  ES=150000
  [ $N -lt $ES ] && ES=$N
  # scale max epochs so big bins get proportionally fewer
  EP=$((200 * 150000 / N))
  [ $EP -lt 30 ] && EP=30
  [ $EP -gt 400 ] && EP=400
  echo "=== R4 $E: $N pos, $EP epochs, epoch_size=$ES $(date -u) ==="
  python3 train.py "$BIN" \
    --validation-size 30000 \
    --check-val-every-n-epoch 1 \
    --epoch-size $ES \
    --batch-size 8192 \
    --max-epochs $EP \
    --default-root-dir $R/runs4/$E \
    >> $R/runs4_$E.trainlog 2>&1

  BEST_EP=$(grep "val_loss_epoch" $R/runs4_$E.trainlog | \
    sed "s/Epoch \([0-9]*\).*val_loss_epoch=\([0-9.]*\).*/\2 \1/" | \
    sort -n | head -1 | awk "{print \$2}")
  CK=$(ls $R/runs4/$E/lightning_logs/version_*/checkpoints/epoch=*-step=*.ckpt 2>/dev/null | \
    sed "s/.*epoch=\([0-9]*\)-.*/\1 &/" | sort -n | \
    awk -v b="$BEST_EP" '$1<=b {c=$2} END{print c}')
  [ -z "$CK" ] && CK=$(ls $R/runs4/$E/lightning_logs/version_*/checkpoints/last.ckpt | head -1)
  if [ -n "$CK" ]; then
    python3 serialize.py "$CK" "$R/nets4/$E.nnue" 2>/dev/null
    echo "=== R4 $E serialized (best epoch $BEST_EP) ==="
  fi
done
touch $R/ROUND4_BIG_DONE
