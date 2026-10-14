#!/bin/bash
# general2: single net on ALL corrected data (clean labels, no MoE routing).
# Fresh, full epochs, plateau-stop. No monitor — Lightning keeps best ckpt.
cd /srv/workspace/flychess/src/nnue-pytorch
B=/srv/workspace/chess-active/bins-general2
R=/srv/workspace/chess-active/runs-general2
mkdir -p $R
TR=$(( $(stat -c %s $B/general2.train.bin) / 40 ))
echo "[$(date +%H:%M)] general2: $TR records, fresh net, clean labels, no monitor"
python3 train.py $B/general2.train.bin --validation-datasets $B/general2.val.bin \
  --validation-size 30000 --check-val-every-n-epoch 1 --epoch-size $TR \
  --batch-size 4096 --max-time 00:48:00:00 --max-epochs 100000 \
  --network-save-period 1 --save_top_k 1 \
  --random-fen-skipping 3 --default-root-dir $R >> $R/train.log 2>&1
echo "[$(date +%H:%M)] general2 trainer exited"
CK=$(python3 -c "
import glob, re
cks = glob.glob('$R/lightning_logs/version_*/checkpoints/epoch=*.ckpt')
def ep(f):
    m = re.search(r'epoch=(\d+)', f); return int(m.group(1)) if m else -1
cks.sort(key=ep); print(cks[-1] if cks else '')")
[ -n "$CK" ] && python3 serialize.py "$CK" $R/best.nnue >/dev/null 2>&1
echo "[$(date +%H:%M)] general2 DONE"
