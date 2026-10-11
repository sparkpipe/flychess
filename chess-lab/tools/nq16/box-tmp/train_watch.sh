#!/bin/bash
# Training-path validation watcher:
#  - serializes every epoch checkpoint to .nnue as it appears
#  - copies last.ckpt -> checkpoint.<epoch>.ckpt every CHECKPOINT_INTERVAL_SEC
CHECKPOINT_INTERVAL_SEC=600   # operator: #define value (hourly for real runs)
RUN=/extnvme/active/train16_test_oppb
NNUE=/srv/workspace/flychess/src/nnue-pytorch
mkdir -p $RUN/nets
cd $NNUE
END=$(( $(date +%s) + 2100 ))   # 35 min of watching (30 train + margin)
LASTCK=0
while [ $(date +%s) -lt $END ]; do
  ck=$(ls -t $RUN/lightning_logs/version_0/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
  if [ -n "$ck" ] && [ "$ck" != "$LASTCK" ]; then
    ep=$(basename "$ck"); ep=${ep#epoch=}; ep=${ep%%-*}
    python3 serialize.py "$ck" $RUN/nets/oppb_e$ep.nnue >> $RUN/watch.log 2>&1 \
      && echo "$(date +%H:%M) exported epoch $ep" >> $RUN/watch.log
    LASTCK=$ck
  fi
  now=$(date +%s)
  if [ $(( now - LASTCK_TIME_ )) -ge $CHECKPOINT_INTERVAL_SEC ] 2>/dev/null || [ -z "$LASTCK_TIME_" ]; then
    if [ -n "$ck" ]; then
      ep=$(basename "$ck"); ep=${ep#epoch=}; ep=${ep%%-*}
      cp "$ck" $RUN/checkpoint.$ep.ckpt
      echo "$(date +%H:%M) wallclock checkpoint.$ep" >> $RUN/watch.log
      LASTCK_TIME_=$now
    else
      LASTCK_TIME_=$now
    fi
  fi
  sleep 10
done
echo "$(date +%H:%M) watcher done" >> $RUN/watch.log
