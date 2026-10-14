#!/bin/bash
# every-ply d1..d12 on the ENTIRE piece_down set, EXACT runs-10min protocol
# (full-size epochs, val 30k, 10-min cap, last.ckpt -> final.nnue).
# Only difference from the historical runs: the label bins (per-depth).
cd /srv/workspace/flychess/src/nnue-pytorch
B=/srv/workspace/chess-active/pd-depthexp/bins
R=/srv/workspace/chess-active/pd-depthexp/runs
for D in 1 2 3 4 5 6 7 8 9 10 11 12; do
  mkdir -p $R/d$D
  TR=$(( $(stat -c %s $B/d$D.train.bin) / 40 ))
  VR=$(( $(stat -c %s $B/d$D.val.bin) / 40 )); [ $VR -gt 30000 ] && VR=30000
  echo "[$(date +%H:%M)] d$D start (epoch-size $TR, val $VR)"
  python3 train.py $B/d$D.train.bin --validation-datasets $B/d$D.val.bin \
    --validation-size $VR --check-val-every-n-epoch 1 --epoch-size $TR \
    --batch-size 4096 --max-time 00:00:10:00 --max-epochs 100000 \
    --network-save-period 100000 --random-fen-skipping 3 \
    --default-root-dir $R/d$D >> $R/d$D/train.log 2>&1
  ck=$(ls -t $R/d$D/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
  [ -n "$ck" ] && python3 serialize.py "$ck" $R/d$D/final.nnue >/dev/null 2>&1
  V=$(python3 -c "
import csv, glob
rows = [r for r in csv.DictReader(open(sorted(glob.glob('$R/d$D/lightning_logs/version_*/metrics.csv'))[-1])) if r.get('val_corr')]
print(max((float(r['val_corr']) for r in rows), default=0))")
  echo "d$D RESULT corr=$V"
done
echo ALL-12-DONE
