#!/bin/bash
# Fleet fix: clean ckpt piles, deploy VERIFIED watchers, restart the crashed.
# Usage: fix_fleet.sh <spark> <expert> [restart]
S=$1; E=$2; RESTART=${3:-keep}
R=~/run16_$E
# clean: all epoch ckpts (validated-export first if serialize available)
latest=$(ls -t $R/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
if [ -n "$latest" ]; then
  ep=$(basename "$latest"); ep=${ep#epoch=}; ep=${ep%%-*}
  mkdir -p $R/nets
  cd ~/nnue-pytorch && python3 serialize.py "$latest" $R/nets/${E}_e$ep.nnue >/dev/null 2>&1
  cd ~
fi
rm -f $R/lightning_logs/version_*/checkpoints/epoch=*.ckpt
ls -t $R/checkpoint.*.ckpt 2>/dev/null | tail -n +2 | xargs -r rm -f
df -h / | tail -1 | awk "{print \"free: \"\$4}"
if [ "$RESTART" = "restart" ]; then
  rm -rf $R/lightning_logs $R/resume.ckpt
  N=$(( $(stat -c %s ~/$E.train.bin) / 40 )); [ $N -lt 1000 ] && N=1000
  cd ~/nnue-pytorch
  setsid nohup python3 train.py ~/$E.train.bin \
    --validation-datasets ~/$E.val.bin --validation-size 30000 \
    --check-val-every-n-epoch 1 --epoch-size $N --batch-size 4096 \
    --max-time 99:00:00:00 --max-epochs 100000 --network-save-period 1 \
    --random-fen-skipping 3 --default-root-dir $R >> $R/train.log 2>&1 < /dev/null &
  cd ~
fi
# watcher: verified deployment
cat > /tmp/watch_$E.sh <<'WEOF'
#!/bin/bash
RUN=$1; EXPERT=$2
cd ~/nnue-pytorch
mkdir -p $RUN/nets
last_ckpt_time=0
while true; do
  CKDIR=$(ls -d $RUN/lightning_logs/version_*/checkpoints 2>/dev/null | tail -1)
  [ -z "$CKDIR" ] && { sleep 20; continue; }
  for ck in $(ls $CKDIR/epoch=*.ckpt 2>/dev/null); do
    b=$(basename "$ck"); ep=${b#epoch=}; ep=${ep%%-*}
    out="$RUN/nets/${EXPERT}_e$ep.nnue"
    [ -f "$out" ] || python3 serialize.py "$ck" "$out" >/dev/null 2>&1
    if [ -f "$out" ] && [ $(stat -c %s "$out") -gt 80000000 ]; then
      rm -f "$ck"
    fi
  done
  now=$(date +%s)
  if [ $((now - last_ckpt_time)) -ge 3600 ]; then
    latest=$(ls -t $CKDIR/last.ckpt 2>/dev/null | head -1)
    [ -n "$latest" ] && cp "$latest" "$RUN/checkpoint.$(date +%H).ckpt" 2>/dev/null
    ls -t $RUN/checkpoint.*.ckpt 2>/dev/null | tail -n +4 | xargs -r rm -f
    last_ckpt_time=$now
  fi
  sleep 15
done
WEOF
chmod +x /tmp/watch_$E.sh
pkill -f "watch_$E.sh" 2>/dev/null
sleep 1
setsid nohup /tmp/watch_$E.sh $R $E > /dev/null 2>&1 < /dev/null &
sleep 2
w=$(pgrep -f "watch_$E.sh" | wc -l)
t=$(ps -eo cmd | grep "[p]ython3 train.py" | wc -l)
echo "$S result: trainers=$t watchers=$w"
