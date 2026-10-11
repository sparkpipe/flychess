#!/bin/bash
# Spark-side watcher: per-epoch .nnue export + DELETE of the 1.9GB ckpt; hourly wallclock.
RUN=$1; EXPERT=$2
CHECKPOINT_INTERVAL_SEC=$3
END=$(( $(date +%s) + $4 ))
CKDIR=$(ls -d $RUN/lightning_logs/version_*/checkpoints 2>/dev/null | tail -1)
cd ~/nnue-pytorch
mkdir -p $RUN/nets
last_ep=-1
last_ckpt_time=0
while [ $(date +%s) -lt $END ]; do
  for ck in $(ls $CKDIR/epoch=*.ckpt 2>/dev/null); do
    b=$(basename "$ck")
    ep=${b#epoch=}; ep=${ep%%-*}
    out="$RUN/nets/${EXPERT}_e$ep.nnue"
    [ -f "$out" ] || python3 serialize.py "$ck" "$out" > /dev/null 2>&1
    if [ -f "$out" ] && [ $(stat -c %s "$out") -gt 80000000 ]; then
      rm -f "$ck"
    fi
    [ "$ep" -gt "$last_ep" ] 2>/dev/null && last_ep=$ep
  done
  now=$(date +%s)
  if [ $((now - last_ckpt_time)) -ge $CHECKPOINT_INTERVAL_SEC ] && [ "$last_ep" -ge 0 ] 2>/dev/null; then
    latest=$(ls -t $CKDIR/epoch=*.ckpt $CKDIR/last.ckpt 2>/dev/null | head -1)
    cp "$latest" "$RUN/checkpoint.$last_ep.ckpt" 2>/dev/null
    # keep only the newest 3 wallclock ckpts (disk safety)
    ls -t $RUN/checkpoint.*.ckpt 2>/dev/null | tail -n +4 | xargs -r rm -f
    last_ckpt_time=$now
  fi
  sleep 15
done
