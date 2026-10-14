#!/bin/bash
# After round-1 finishes: archive every best.nnue (sha256 receipt), then
# CONTINUE training from the best checkpoint — open-ended, plateau-stop only.
# Round-2 never deletes round-1 artifacts; it writes to runs-correct16-r2/.
cd /srv/workspace/chess-active
echo "[$(date +%H:%M)] waiting for round-1 queue..."
while ! grep -q "CORRECT16-QUEUE-DONE" correct16.log 2>/dev/null; do
  sleep 300
done
echo "[$(date +%H:%M)] round-1 done — archiving bests"
A=/mnt/cold-raid6/chess-audit/training-set-archive/correct16-round1
mkdir -p $A
for e in piece_down dv_Q tb oppb mvr dv_R qvmat rv2m op_pawnimb mg_unsafe mg_safe nvb dv_rest op_even_l0 op_even_l1 op_even_l2p; do
  [ -f runs-correct16/$e/best.nnue ] && cp runs-correct16/$e/best.nnue $A/$e.nnue
done
(cd $A && sha256sum *.nnue > SHA256SUMS && ls | wc -l)

echo "[$(date +%H:%M)] round-2: continue from best checkpoints, open-ended"
cd /srv/workspace/flychess/src/nnue-pytorch
B=/srv/workspace/chess-active/bins-correct16
R1=/srv/workspace/chess-active/runs-correct16
R2=/srv/workspace/chess-active/runs-correct16-r2
mkdir -p $R2
for e in piece_down dv_Q tb oppb mvr dv_R qvmat rv2m op_pawnimb mg_unsafe mg_safe nvb dv_rest op_even_l0 op_even_l1 op_even_l2p; do
  mkdir -p $R2/$e
  TR=$(( $(stat -c %s $B/$e.train.bin) / 40 ))
  VR=$(( $(stat -c %s $B/$e.val.bin) / 40 )); [ $VR -gt 30000 ] && VR=30000
  # resume from round-1's newest intact ckpt
  SEED=$(python3 - <<EOF
import glob, re, zipfile
cks = glob.glob("$R1/$e/lightning_logs/version_*/checkpoints/epoch=*.ckpt")
def ep(f):
    m = re.search(r"epoch=(\d+)", f)
    return int(m.group(1)) if m else -1
cks.sort(key=ep, reverse=True)
for c in cks:
    try:
        zipfile.ZipFile(c).namelist()
        print(c); break
    except Exception:
        continue
EOF
)
  RESUME=""
  [ -n "$SEED" ] && RESUME="--resume-from-checkpoint $SEED"
  echo "[$(date +%H:%M)] $e r2 start (epoch-size $TR, resume=${SEED##*/})"
  python3 train.py $B/$e.train.bin --validation-datasets $B/$e.val.bin \
    --validation-size $VR --check-val-every-n-epoch 1 --epoch-size $TR \
    --batch-size 4096 --max-time 00:48:00:00 --max-epochs 100000 \
    --network-save-period 1 --save_top_k 1 \
    $RESUME \
    --random-fen-skipping 3 --default-root-dir $R2/$e >> $R2/$e/train.log 2>&1 &
  TPID=$!
  BEST=-1; BEST_EP=-1
  while kill -0 $TPID 2>/dev/null; do
    sleep 120
    M=$(ls -t $R2/$e/lightning_logs/version_*/metrics.csv 2>/dev/null | head -1)
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
      if python3 -c "exit(0 if $B2 > $BEST else 1)" 2>/dev/null; then BEST=$B2; BEST_EP=$E2; fi
      GAP=$(( L2 - BEST_EP ))
      if [ $GAP -ge 40 ]; then
        echo "[$(date +%H:%M)] $e r2 plateau (best $BEST @e$BEST_EP) — stop"
        kill $TPID 2>/dev/null
        for w in 1 2 3 4 5 6; do sleep 10; kill -0 $TPID 2>/dev/null || break; done
        kill -9 $TPID 2>/dev/null
        break
      fi
    fi
    find $R2/$e/lightning_logs -name "epoch=*.ckpt" | sort | head -n -3 | xargs -r rm -f
    DU=$(du -sm $R2/$e 2>/dev/null | cut -f1)
    [ "$DU" -gt 12000 ] && { echo "[$(date +%H:%M)] $e r2 DISK GUARD"; kill $TPID; break; }
  done
  wait $TPID 2>/dev/null
  CK=$(python3 - <<EOF
import glob, re, zipfile
cks = glob.glob("$R2/$e/lightning_logs/version_*/checkpoints/epoch=*.ckpt")
def ep(f):
    m = re.search(r"epoch=(\d+)", f)
    return int(m.group(1)) if m else -1
cks.sort(key=ep, reverse=True)
for c in cks:
    try:
        zipfile.ZipFile(c).namelist()
        print(c); break
    except Exception:
        continue
EOF
)
  [ -n "$CK" ] && python3 serialize.py "$CK" $R2/$e/best.nnue >/dev/null 2>&1
  # keep round-1 best if round-2 didn't beat its own start
  echo "$e R2 DONE: best $BEST @e$BEST_EP"
done
echo ROUND2-DONE
