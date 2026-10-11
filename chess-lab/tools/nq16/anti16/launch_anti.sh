#!/bin/bash
# Launch ONE anti16 expert training run on this host.
# $1 = expert name. Env overrides: BINDIR (bins), NNUEDIR (trainer), WATCH (watcher script).
E=$1
[ -z "$E" ] && { echo "usage: launch_anti.sh <expert>"; exit 1; }
BINDIR=${BINDIR:-$HOME}
NNUEDIR=${NNUEDIR:-$HOME/nnue-pytorch}
WATCH=${WATCH:-/tmp/anti_watch_spark.sh}
T=${TRAINBIN:-$BINDIR/anti_$E.train.bin}
V=${VALBIN:-$BINDIR/anti_$E.val.bin}
[ -f "$T" ] || { echo "missing $T"; exit 1; }
[ -f "$V" ] || { echo "missing $V"; exit 1; }
[ -f "$NNUEDIR/train.py" ] || { echo "missing $NNUEDIR/train.py"; exit 1; }
TR=$(( $(stat -c %s "$T") / 40 ))
VR=$(( $(stat -c %s "$V") / 40 ))
[ "$VR" -gt 30000 ] && VR=30000
R=${RUNDIR:-$HOME/runanti_$E}
mkdir -p "$R"
cd "$NNUEDIR" || exit 1
if pgrep -f "train\.py .*runanti_$E" >/dev/null; then
  echo "trainer for $E already running, skip"
  exit 0
fi
nohup python3 train.py "$T" \
  --validation-datasets "$V" \
  --validation-size "$VR" \
  --check-val-every-n-epoch 1 \
  --epoch-size "$TR" \
  --batch-size 4096 \
  --max-time 99:00:00:00 \
  --max-epochs 100000 \
  --network-save-period 1 \
  --random-fen-skipping 3 \
  --default-root-dir "$R" >> "$R/train.log" 2>&1 < /dev/null &
echo "trainer pid $! (epoch-size $TR, val $VR)"
sleep 2
nohup bash "$WATCH" "$R" "$E" 3600 9999999 >> "$R/watch_nohup.log" 2>&1 < /dev/null &
echo "watcher pid $!"
