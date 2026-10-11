#!/bin/bash
CHECKPOINT_INTERVAL_SEC=600
RUN=/extnvme/active/train16_test_oppb
CKDIR=$RUN/lightning_logs/version_0/checkpoints
NNUE=/srv/workspace/flychess/src/nnue-pytorch
mkdir -p $RUN/nets
cd $NNUE
last_ep=-1
last_ckpt_time=0
END=$(( $(date +%s) + 2100 ))
while [ $(date +%s) -lt $END ]; do
  ck=$(ls -t $CKDIR/epoch=*.ckpt 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    b=$(basename "$ck")
    ep=${b#epoch=}; ep=${ep%%-*}
    if [ "$ep" -gt "$last_ep" ] 2>/dev/null; then
      python3 serialize.py "$ck" "$RUN/nets/oppb_e$ep.nnue" >> $RUN/watch.log 2>&1 \
        && echo "$(date +%H:%M) nnue epoch $ep" >> $RUN/watch.log
      last_ep=$ep
    fi
    now=$(date +%s)
    if [ $((now - last_ckpt_time)) -ge $CHECKPOINT_INTERVAL_SEC ]; then
      cp "$ck" "$RUN/checkpoint.$ep.ckpt"
      echo "$(date +%H:%M) wallclock checkpoint.$ep" >> $RUN/watch.log
      last_ckpt_time=$now
    fi
  fi
  sleep 10
done
echo "$(date +%H:%M) watcher done" >> $RUN/watch.log
