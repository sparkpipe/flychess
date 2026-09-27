#!/bin/bash
# 7-way MoE training: each bucket trains for equal time
# (total = same as the single-net baseline: ~16 min)
# Runs sequentially on the 5090 (each ~2.3 min = ~14 epochs of 1M)
set -u
BUCKETS=/home/spec/chess-lab/buckets
OUT=/home/spec/chess-lab/nnue_7way
LOG=/home/spec/chess-lab/train_7way.log
mkdir -p $OUT
cd /home/spec/nnue-pytorch

echo "=== 7-way MoE training: equal budget per bucket ===" >> $LOG
echo "=== started $(date) ===" >> $LOG

for B in 0 1 2 3 4 5 6; do
    DATA=$BUCKETS/bucket_${B}.bin
    if [ ! -f "$DATA" ]; then
        echo "bucket $B: MISSING" >> $LOG
        continue
    fi
    SIZE=$(stat -c %s $DATA)
    ROWS=$((SIZE / 40))
    EPOCH_SIZE=$((ROWS < 1000000 ? ROWS : 1000000))
    N_EPOCHS=14  # ~2.3 min at ~50it/s with batch 1024
    echo "bucket $B: $ROWS rows, $N_EPOCHS epochs x $EPOCH_SIZE" >> $LOG

    python3 train.py \
        $DATA \
        --max-epochs $N_EPOCHS \
        --batch-size 1024 \
        --epoch-size $EPOCH_SIZE \
        --default-root-dir $OUT/bucket_${B} \
        --threads 8 \
        --seed $((42 + B)) \
        --save-last-network True \
        >> $LOG 2>&1

    # serialize each bucket's net
    CKPT=$(find $OUT/bucket_${B} -name "last.ckpt" | head -1)
    if [ -n "$CKPT" ]; then
        python3 serialize.py \
            $CKPT \
            $OUT/bucket_${B}.nnue 2>> $LOG
        echo "bucket $B: serialized to $OUT/bucket_${B}.nnue" >> $LOG
    fi
done

echo "=== ALL 7 BUCKETS TRAINED $(date) ===" >> $LOG
echo "7WAY-TRAINING-COMPLETE" >> $LOG
