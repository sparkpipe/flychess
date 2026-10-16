#!/bin/bash
# wait for train.py to finish, then serialize the best checkpoint to best.nnue
: "${CHESS_DATA:?CHESS_DATA not set (data root)}"
R=$CHESS_DATA/runs-general1
while pgrep -f "train.py.*general1" > /dev/null; do sleep 60; done
CK=$(python3 -c "
import glob, re
cks = glob.glob(\"$R/lightning_logs/version_*/checkpoints/epoch=*.ckpt\")
def ep(f):
    m = re.search(r\"epoch=(\d+)\", f); return int(m.group(1)) if m else -1
cks.sort(key=ep); print(cks[-1] if cks else \"\")")
[ -n "$CK" ] && cd REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO/nnue-pytorch" && python3 serialize.py "$CK" $R/best.nnue && echo "$(date +%H:%M) serialized $(basename $CK) -> best.nnue"
