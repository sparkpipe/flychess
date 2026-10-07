#!/bin/bash
# Spark rescue: clean fat ckpts (keep last.ckpt), resume training, start watcher.
E=$1
R=~/run16_$E
NNUE=~/nnue-pytorch
cd $NNUE
# export the LAST epoch ckpt before deleting the pile (for the record), keep last.ckpt
latest_ck=$(ls -t $R/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
if [ -n "$latest_ck" ]; then
  ep=$(basename "$latest_ck"); ep=${ep#epoch=}; ep=${ep%%-*}
  python3 serialize.py "$latest_ck" $R/net_e$ep.nnue >/dev/null 2>&1
fi
rm -f $R/lightning_logs/version_*/checkpoints/epoch=*.ckpt
rm -f $R/checkpoint.*.ckpt
df -h / | tail -1
cp $R/lightning_logs/version_*/checkpoints/last.ckpt $R/resume.ckpt 2>/dev/null
N=$(( $(stat -c %s ~/$E.train.bin) / 40 )); [ $N -lt 1000 ] && N=1000
setsid nohup python3 train.py ~/$E.train.bin \
  --validation-datasets ~/$E.val.bin --validation-size 30000 \
  --check-val-every-n-epoch 1 --epoch-size $N --batch-size 4096 \
  --max-time 99:00:00:00 --max-epochs 100000 --network-save-period 1 \
  --random-fen-skipping 3 \
  --resume-from-checkpoint $R/resume.ckpt \
  --default-root-dir $R >> $R/train.log 2>&1 < /dev/null &
sleep 3
echo "resumed: $(ps -eo cmd | grep '[p]ython3 train.py' | wc -l) trainer(s)"
# watcher: export+delete every epoch ckpt, hourly wallclock
setsid nohup /tmp/train_watch_spark.sh $R $E 3600 9999999 > /dev/null 2>&1 < /dev/null &
echo "watcher started"
