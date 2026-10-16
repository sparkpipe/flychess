#!/bin/bash
# general3C: curriculum training d10 -> d11 -> ... -> d20.
# Each stage runs to PLATEAU (no improvement for 15 validations).
# Warm from previous stage's last checkpoint.
cd /srv/workspace/flychess/src/nnue-pytorch
B=/srv/workspace/chess-active/bins-curriculum
R=/srv/workspace/chess-active/runs-general3c
mkdir -p $R
PREV_CKPT=""
for D in 10 11 12 13 14 15 16 17 18 19 20; do
  SR=$R/stage_d$D
  mkdir -p $SR
  TR=$(( $(stat -c %s $B/d$D/general3c.train.bin) / 40 ))
  VR=$(( $(stat -c %s $B/d$D/general3c.val.bin) / 40 ))
  [ $VR -gt 50000 ] && VR=50000
  RESUME=""
  [ -n "$PREV_CKPT" ] && RESUME="--resume-from-checkpoint $PREV_CKPT"
  echo "[$(date +%H:%M)] stage d$D: $TR recs, plateau-stop (15 stale), resume=${PREV_CKPT:+yes}"
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
  # get best intact ckpt from this stage
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
