#!/bin/bash
# Continue training the quick-16 from their latest checkpoints (NOT from scratch).
# Sequential, weak experts first, plateau stop, best.nnue kept, disk-guarded.
set -u
cd /srv/workspace/flychess/src/nnue-pytorch
B=/srv/workspace/chess-active/bins-db-v1
R=/srv/workspace/chess-active/runs-continue
Q1=/srv/workspace/chess-active/runs-10min
Q2=/srv/workspace/chess-active/runs-extra10
mkdir -p $R
ORDER="piece_down nvb dv_Q dv_R mvr oppb tb qvmat rv2m op_pawnimb mg_unsafe mg_safe dv_rest op_even_l0 op_even_l1 op_even_l2p"
for e in $ORDER; do
  CK=$(ls -t $Q2/$e/lightning_logs/version_*/checkpoints/last.ckpt $Q1/$e/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
  [ -z "$CK" ] && { echo "$e: no checkpoint to resume — skip"; continue; }
  RUN=$R/$e
  mkdir -p $RUN
  TR=$(( $(stat -c %s $B/$e.train.bin) / 40 ))
  VR=$(( $(stat -c %s $B/$e.val.bin) / 40 )); [ $VR -gt 30000 ] && VR=30000
  echo "[$(date +%H:%M)] $e: resuming from $CK (epoch-size $TR)"
  python3 train.py $B/$e.train.bin --validation-datasets $B/$e.val.bin \
    --validation-size $VR --check-val-every-n-epoch 1 --epoch-size $TR \
    --batch-size 4096 --max-time 00:12:00:00 --max-epochs 100000 \
    --network-save-period 1 --save_top_k 1 \
    --resume-from-checkpoint "$CK" \
    --random-fen-skipping 3 --default-root-dir $RUN >> $RUN/train.log 2>&1 &
  TPID=$!
  BEST=-1; BEST_EP=-1; LAST_EP=-1
  while kill -0 $TPID 2>/dev/null; do
    sleep 120
    M=$(ls -t $RUN/lightning_logs/version_*/metrics.csv 2>/dev/null | head -1)
    [ -z "$M" ] && continue
    read B2 E2 L2 < <(python3 - "$M" <<'EOF'
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
EOF
)
    if [ -n "${B2:-}" ] && [ "$B2" != "-1" ]; then
      LAST_EP=$L2
      if python3 -c "exit(0 if $B2 > $BEST else 1)" 2>/dev/null; then BEST=$B2; BEST_EP=$E2; fi
      GAP=$(( LAST_EP - BEST_EP ))
      if [ $GAP -ge 40 ]; then
        echo "[$(date +%H:%M)] $e: plateau (best $BEST @e$BEST_EP, now e$LAST_EP) — stop"
        kill $TPID; sleep 8; kill -9 $TPID 2>/dev/null; break
      fi
    fi
    find $RUN/lightning_logs -name "epoch=*.ckpt" | sort | head -n -3 | xargs -r rm -f
    DU=$(du -sm $RUN 2>/dev/null | cut -f1)
    if [ "$DU" -gt 12000 ]; then echo "[$(date +%H:%M)] $e: DISK GUARD ${DU}MB — stop"; kill $TPID; break; fi
  done
  wait $TPID 2>/dev/null
  CK2=$(python3 -c "
import glob, re
cks = glob.glob('$RUN/lightning_logs/version_*/checkpoints/epoch=*.ckpt')
def ep(f):
    m = re.search(r'epoch=(\d+)', f); return int(m.group(1)) if m else -1
cks.sort(key=ep); print(cks[-1] if cks else '')")
  [ -n "$CK2" ] && python3 serialize.py "$CK2" $RUN/best.nnue >/dev/null 2>&1 && echo "$e DONE: best $BEST @e$BEST_EP -> best.nnue"
done
echo CONTINUE-QUEUE-DONE
