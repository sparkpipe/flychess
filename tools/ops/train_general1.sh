#!/bin/bash
# SINGLE GENERAL NET: all 16 experts positions in ONE net (no routing) — baseline vs n16 MoE.
set -u
cd /srv/workspace/flychess/src/nnue-pytorch
B=/srv/workspace/chess-active/bins-general
R=/srv/workspace/chess-active/runs-general1
mkdir -p $R
[ -f $R/best.nnue ] && { echo "general1: done"; exit 0; }
echo "[$(date +%H:%M)] general1: 15.3M recs, fresh net, 6h cap, plateau 40ep"
python3 train.py $B/general1.train.bin --validation-datasets $B/general1.val.bin \
  --validation-size 300000 --check-val-every-n-epoch 1 --epoch-size 4000000 \
  --batch-size 4096 --max-time 00:06:00:00 --max-epochs 100000 \
  --network-save-period 1 --save_top_k 1 \
  --random-fen-skipping 3 --default-root-dir $R >> $R/train.log 2>&1 &
TPID=$!
BEST=-1; BEST_EP=-1
while kill -0 $TPID 2>/dev/null; do
  sleep 120
  M=$(ls -t $R/lightning_logs/version_*/metrics.csv 2>/dev/null | head -1)
  [ -z "$M" ] && continue
  read B2 E2 L2 < <(python3 - "$M" << "PYEOF"
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
    if [ $GAP -ge 40 ]; then
      echo "[$(date +%H:%M)] general1: plateau (best $BEST @e$BEST_EP, now e$L2) — stop"
      kill $TPID; sleep 8; kill -9 $TPID 2>/dev/null; break
    fi
  fi
  find $R/lightning_logs -name "epoch=*.ckpt" | sort | head -n -3 | xargs -r rm -f
  DU=$(du -sm $R 2>/dev/null | cut -f1)
  if [ "$DU" -gt 30000 ]; then echo "[$(date +%H:%M)] general1: DISK GUARD ${DU}MB"; kill $TPID; break; fi
done
wait $TPID 2>/dev/null
CK2=$(python3 -c "
import glob, re
cks = glob.glob(\"$R/lightning_logs/version_*/checkpoints/epoch=*.ckpt\")
def ep(f):
    m = re.search(r\"epoch=(\d+)\", f); return int(m.group(1)) if m else -1
cks.sort(key=ep); print(cks[-1] if cks else \"\")")
[ -n "$CK2" ] && python3 serialize.py "$CK2" $R/best.nnue >/dev/null 2>&1 && echo "general1 DONE: best $BEST @e$BEST_EP"
