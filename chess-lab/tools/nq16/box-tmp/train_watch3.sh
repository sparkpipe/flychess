#!/bin/bash
# Export-then-delete policy (operator 2026-10-05):
#   - every epoch ckpt -> .nnue (keep), then DELETE the 1.86GB ckpt
#   - full checkpoint.N copy every CHECKPOINT_INTERVAL_SEC (define)
#   - last.ckpt stays (Lightning rolling resume, single file)
CHECKPOINT_INTERVAL_SEC=600
RUN=/extnvme/active/train16_test_oppb
CKDIR=$RUN/lightning_logs/version_0/checkpoints
NNUE=/srv/workspace/flychess/src/nnue-pytorch
mkdir -p $RUN/nets
cd $NNUE
last_ep=-1
last_ckpt_time=0
END=$(( $(date +%s) + 1500 ))
while [ $(date +%s) -lt $END ]; do
  for ck in $(ls $CKDIR/epoch=*.ckpt 2>/dev/null); do
    b=$(basename "$ck")
    ep=${b#epoch=}; ep=${ep%%-*}
    out="$RUN/nets/oppb_e$ep.nnue"
    if [ ! -f "$out" ]; then
      python3 serialize.py "$ck" "$out" >> $RUN/watch.log 2>&1
    fi
    if [ -f "$out" ] && [ $(stat -c %s "$out") -gt 80000000 ]; then
      rm -f "$ck"
      echo "$(date +%H:%M) exported+deleted epoch $ep" >> $RUN/watch.log
    fi
    last_ep=$ep
  done
  now=$(date +%s)
  if [ $((now - last_ckpt_time)) -ge $CHECKPOINT_INTERVAL_SEC ] && [ "$last_ep" -ge 0 ]; then
    latest=$(ls -t $CKDIR/epoch=*.ckpt $CKDIR/last.ckpt 2>/dev/null | head -1)
    cp "$latest" "$RUN/checkpoint.$last_ep.ckpt"
    echo "$(date +%H:%M) wallclock checkpoint.$last_ep" >> $RUN/watch.log
    last_ckpt_time=$now
  fi
  sleep 10
done
echo "$(date +%H:%M) watcher done" >> $RUN/watch.log
