#!/bin/bash
# Generic watcher: per-epoch .nnue export+delete, wallclock checkpoints.
# $1 = run dir, $2 = expert name, $3 = checkpoint interval sec, $4 = watch seconds
RUN=$1; EXPERT=$2
CHECKPOINT_INTERVAL_SEC=$3
END=$(( $(date +%s) + $4 ))
CKDIR=$(ls -d $RUN/lightning_logs/version_*/checkpoints 2>/dev/null | tail -1)
NNUE=/srv/workspace/flychess/src/nnue-pytorch
mkdir -p $RUN/nets
cd $NNUE
last_ep=-1
last_ckpt_time=0
while [ $(date +%s) -lt $END ]; do
  for ck in $(ls $CKDIR/epoch=*.ckpt 2>/dev/null); do
    b=$(basename "$ck")
    ep=${b#epoch=}; ep=${ep%%-*}
    out="$RUN/nets/${EXPERT}_e$ep.nnue"
    [ -f "$out" ] || python3 serialize.py "$ck" "$out" >> $RUN/watch.log 2>&1
    if [ -f "$out" ] && [ $(stat -c %s "$out") -gt 80000000 ]; then
      rm -f "$ck"
      echo "$(date +%H:%M) exported+deleted epoch $ep" >> $RUN/watch.log
    fi
    [ "$ep" -gt "$last_ep" ] && last_ep=$ep
  done
  now=$(date +%s)
  if [ $((now - last_ckpt_time)) -ge $CHECKPOINT_INTERVAL_SEC ] && [ "$last_ep" -ge 0 ]; then
    latest=$(ls -t $CKDIR/epoch=*.ckpt 2>/dev/null | head -1)
    [ -z "$latest" ] && latest=$CKDIR/last.ckpt
    cp "$latest" "$RUN/checkpoint.$last_ep.ckpt"
    echo "$(date +%H:%M) wallclock checkpoint.$last_ep" >> $RUN/watch.log
    last_ckpt_time=$now
  fi
  sleep 10
done
echo "$(date +%H:%M) watcher done" >> $RUN/watch.log
