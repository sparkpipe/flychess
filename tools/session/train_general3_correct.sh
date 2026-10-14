#!/bin/bash
# general3 (correct set): warm-start from general2's best checkpoint.
# Training set: OTB/puzzle d20 + clean short wins + gambit duplication
# (NO unfiltered miniature noise).
cd /srv/workspace/chess-active
# wait for build to finish
while ! grep -q "TRAIN:" build_g3v2.log 2>/dev/null; do
  sleep 30
done
tail -3 build_g3v2.log

cd /srv/workspace/flychess/src/nnue-pytorch
B=/srv/workspace/chess-active/bins-general3
R=/srv/workspace/chess-active/runs-general3
CKPT=/srv/workspace/chess-active/runs-general2/lightning_logs/version_*/checkpoints/epoch=54-step=204545.ckpt
mkdir -p $R
TR=$(( $(stat -c %s $B/general3.train.bin) / 40 ))
echo "[$(date +%H:%M)] general3: $TR records, warm from general2, correct set"
python3 train.py $B/general3.train.bin --validation-datasets $B/general3.val.bin \
  --validation-size 50000 --check-val-every-n-epoch 1 --epoch-size $TR \
  --batch-size 4096 --max-time 00:48:00:00 --max-epochs 100000 \
  --network-save-period 1 --save_top_k 1 \
  --resume-from-checkpoint $CKPT \
  --random-fen-skipping 3 --default-root-dir $R >> $R/train.log 2>&1
CK=$(python3 -c "
import glob, re
cks = glob.glob('$R/lightning_logs/version_*/checkpoints/epoch=*.ckpt')
def ep(f):
    m = re.search(r'epoch=(\d+)', f); return int(m.group(1)) if m else -1
cks.sort(key=ep); print(cks[-1] if cks else '')")
[ -n "$CK" ] && python3 serialize.py "$CK" $R/best.nnue >/dev/null 2>&1
echo "[$(date +%H:%M)] general3 DONE"
