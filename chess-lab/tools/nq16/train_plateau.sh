#!/bin/bash
# Plateau trainer: one expert at a time; best-epoch ckpt kept by CheckpointManager
# (save_top_k=1); best.nnue serialized at end; plateau stop on val_corr patience;
# disk-guarded (bounded ckpts, no tmp accumulation).
# Usage: train_plateau.sh <expert> <bins_dir> <out_root> <patience_epochs> [max_hours]
set -u
E=$1; B=$2; R=$3; PAT=${4:-40}; MAXH=${5:-12}
cd /srv/workspace/flychess/src/nnue-pytorch
RUN=$R/$E
mkdir -p $RUN
TR=$(( $(stat -c %s $B/$E.train.bin) / 40 ))
VR=$(( $(stat -c %s $B/$E.val.bin) / 40 )); [ $VR -gt 30000 ] && VR=30000
H=$(printf "%02d" $MAXH)
echo "[$(date +%H:%M)] $E: epoch-size $TR val $VR patience $PAT cap ${MAXH}h"
python3 train.py $B/$E.train.bin --validation-datasets $B/$E.val.bin \
  --validation-size $VR --check-val-every-n-epoch 1 --epoch-size $TR \
  --batch-size 4096 --max-time 00:$H:00:00 --max-epochs 100000 \
  --network-save-period 1 --save_top_k 1 \
  --random-fen-skipping 3 --default-root-dir $RUN >> $RUN/train.log 2>&1 &
TPID=$!
BEST=-1; BEST_EP=-1; LAST_EP=-1
while kill -0 $TPID 2>/dev/null; do
  sleep 120
  M=$(ls -t $RUN/lightning_logs/version_*/metrics.csv 2>/dev/null | head -1)
  [ -z "$M" ] && continue
  python3 - "$M" <<'EOF' > /tmp/pl_$$.txt
import csv, sys
best = (-1, -1); last = -1
for r in csv.DictReader(open(sys.argv[1])):
    c = r.get("val_corr")
    if c in (None, ""):
        continue
    try:
        ep, v = int(r["epoch"]), float(c)
    except (ValueError, KeyError):
        continue
    last = ep
    if v > best[0]:
        best = (v, ep)
print(best[0], best[1], last)
EOF
  read B2 E2 L2 < /tmp/pl_$$.txt
  if [ -n "$B2" ] && [ "$B2" != "-1" ]; then
    LAST_EP=$L2
    if python3 -c "exit(0 if $B2 > $BEST else 1)" 2>/dev/null; then
      BEST=$B2; BEST_EP=$E2
    fi
    GAP=$(( LAST_EP - BEST_EP ))
    if [ $GAP -ge $PAT ]; then
      echo "[$(date +%H:%M)] $E: plateau (best $BEST @e$BEST_EP, now e$LAST_EP, gap $GAP) — stopping"
      kill $TPID; sleep 8; kill -9 $TPID 2>/dev/null; break
    fi
  fi
  # disk guard: prune stray epoch ckpts, assert bounded
  find $RUN/lightning_logs -name "epoch=*.ckpt" | sort | head -n -3 | xargs -r rm -f
  DU=$(du -sm $RUN 2>/dev/null | cut -f1)
  [ "$DU" -gt 12000 ] && { echo "[$(date +%H:%M)] $E: DISK GUARD ${DU}MB — stopping"; kill $TPID; break; }
done
wait $TPID 2>/dev/null
CK=$(python3 - <<EOF
import glob, re
cks = glob.glob("$RUN/lightning_logs/version_*/checkpoints/epoch=*.ckpt")
def ep(f):
    m = re.search(r"epoch=(\d+)", f)
    return int(m.group(1)) if m else -1
cks.sort(key=ep)
# CheckpointManager top_k=1: the surviving non-last epoch ckpt IS the best
cands = [c for c in cks]
print(cands[-1] if cands else "")
EOF
)
[ -n "$CK" ] && python3 serialize.py "$CK" $RUN/best.nnue >/dev/null 2>&1 && echo "$E DONE: best corr $BEST @e$BEST_EP -> best.nnue" || echo "$E DONE (no ckpt survived)"
