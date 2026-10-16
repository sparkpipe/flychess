#!/bin/bash
# Resume general3C curriculum after suspension. Continues the current stage
# from its OWN last.ckpt (mid-stage continuation), then chains remaining
# stages exactly like train_general3c.sh (plateau = 15 stale validations,
# 6h per-stage cap, warm from previous stage's best ckpt).
# Usage: resume_general3c.sh [start-depth]   (default: auto-detect)
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO/nnue-pytorch"
: "${CHESS_DATA:?CHESS_DATA not set (data root holding bins-curriculum/ and runs-general3c/)}"
B="$CHESS_DATA/bins-curriculum"
R="$CHESS_DATA/runs-general3c"
START=${1:-auto}
if [ "$START" = auto ]; then
  START=20
  for D in 20 19 18 17 16 15 14 13 12 11 10; do
    ls $R/stage_d$D/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && { START=$D; break; }
  done
fi
echo "resuming general3C at stage d$START"
PREV_CKPT=""
for D in $(seq $START 20); do
  SR=$R/stage_d$D
  mkdir -p $SR
  TR=$(( $(stat -c %s $B/d$D/general3c.train.bin) / 40 ))
  VR=$(( $(stat -c %s $B/d$D/general3c.val.bin) / 40 ))
  [ $VR -gt 50000 ] && VR=50000
  OWN=$(ls -t $SR/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
  if [ -n "$OWN" ]; then
    RESUME="--resume-from-checkpoint $OWN"
    echo "[$(date +%H:%M)] stage d$D: CONTINUING from own last.ckpt"
  elif [ -n "$PREV_CKPT" ]; then
    RESUME="--resume-from-checkpoint $PREV_CKPT"
    echo "[$(date +%H:%M)] stage d$D: warm from previous stage best"
  else
    RESUME=""
    echo "[$(date +%H:%M)] stage d$D: fresh"
  fi
  python3 train.py $B/d$D/general3c.train.bin \
    --validation-datasets $B/d$D/general3c.val.bin \
    --validation-size $VR --check-val-every-n-epoch 1 --epoch-size $TR \
    --batch-size 4096 --max-time 00:06:00:00 --max-epochs 100000 \
    --network-save-period 1 --save_top_k 1 \
    $RESUME \
    --random-fen-skipping 3 --default-root-dir $SR >> $SR/train.log 2>&1 &
  TPID=$!
  BEST=-1; BEST_EP=-1
  while kill -0 $TPID 2>/dev/null; do
    sleep 120
    M=$(ls -t $SR/lightning_logs/version_*/metrics.csv 2>/dev/null | head -1)
    [ -z "$M" ] && continue
    read B2 E2 L2 < <(python3 - "$M" << 'PYEOF'
import csv, sys
best = (-1, -1); last = -1
for r in csv.DictReader(open(sys.argv[1])):
    c = r.get("val_corr")
    if c in (None, ""): continue
    try: ep, v = int(r["epoch"]), float(c)
    except (ValueError, KeyError): continue
    last = ep
    if v > best[0]: best = (v, ep)
print(best[0], best[1], last)
PYEOF
)
    if [ -n "${B2:-}" ] && [ "$B2" != "-1" ]; then
      if python3 -c "exit(0 if $B2 > $BEST else 1)" 2>/dev/null; then BEST=$B2; BEST_EP=$E2; fi
      GAP=$(( L2 - BEST_EP ))
      if [ $GAP -ge 15 ]; then
        echo "[$(date +%H:%M)] d$D: plateau (best $BEST @e$BEST_EP, gap=$GAP) — advancing"
        kill $TPID; sleep 10; kill -9 $TPID 2>/dev/null; break
      fi
    fi
  done
  wait $TPID 2>/dev/null
  PREV_CKPT=$(python3 << EOF
import glob, re, zipfile
cks = glob.glob("$SR/lightning_logs/version_*/checkpoints/epoch=*.ckpt")
def ep(f):
    m = re.search(r"epoch=(\d+)", f); return int(m.group(1)) if m else -1
cks.sort(key=ep, reverse=True)
for c in cks:
    try:
        zipfile.ZipFile(c).namelist(); print(c); break
    except: continue
EOF
)
  [ -z "$PREV_CKPT" ] && { echo "stage d$D FAILED — no ckpt"; exit 1; }
  echo "  d$D done: best corr $BEST @e$BEST_EP"
done
python3 serialize.py "$PREV_CKPT" $R/best.nnue >/dev/null 2>&1
echo "[$(date +%H:%M)] general3C DONE"
