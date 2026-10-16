#!/bin/bash
# general3B: FRESH net on the same correct d20 data — tests whether
# general2's warm start biased general3. Separate output dir, same bins.
: "${CHESS_DATA:?CHESS_DATA not set (data root)}"
cd REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO/nnue-pytorch"
B=$CHESS_DATA/bins-general3
R=$CHESS_DATA/runs-general3b
mkdir -p $R
TR=$(( $(stat -c %s $B/general3.train.bin) / 40 ))
echo "[$(date +%H:%M)] general3B: $TR records, FRESH net (no warm start)"
python3 train.py $B/general3.train.bin --validation-datasets $B/general3.val.bin \
  --validation-size 50000 --check-val-every-n-epoch 1 --epoch-size $TR \
  --batch-size 4096 --max-time 00:48:00:00 --max-epochs 100000 \
  --network-save-period 1 --save_top_k 1 \
  --random-fen-skipping 3 --default-root-dir $R >> $R/train.log 2>&1
CK=$(python3 -c "
import glob, re
cks = glob.glob('$R/lightning_logs/version_*/checkpoints/epoch=*.ckpt')
def ep(f):
    m = re.search(r'epoch=(\d+)', f); return int(m.group(1)) if m else -1
cks.sort(key=ep); print(cks[-1] if cks else '')")
[ -n "$CK" ] && python3 serialize.py "$CK" $R/best.nnue >/dev/null 2>&1
echo "[$(date +%H:%M)] general3B DONE"
