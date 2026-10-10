#!/bin/bash
# Anti engine: 30min/expert from baseline bins, weak-first, plateau-stop.
# Anti training uses the same bins (the anti positions route to the same experts
# — the corpus contains both sides via the seg/puzzle sources).
set -u
cd /srv/workspace/flychess/src/nnue-pytorch
B=/srv/workspace/chess-active/bins-fleet-v1
R=/srv/workspace/chess-active/runs-anti30
Q=/srv/workspace/chess-active/engine16-baseline
mkdir -p $R
ORDER="piece_down dv_Q tb oppb mvr dv_R qvmat rv2m op_pawnimb mg_unsafe mg_safe nvb dv_rest op_even_l0 op_even_l1 op_even_l2p"
for e in $ORDER; do
  [ -f $R/$e/best.nnue ] && { echo "$e: done"; continue; }
  mkdir -p $R/$e
  TR=$(( $(stat -c %s $B/$e.train.bin) / 40 ))
  VR=$(( $(stat -c %s $B/$e.val.bin) / 40 )); [ $VR -gt 30000 ] && VR=30000
  # resume from baseline best (net, not ckpt — different run tree)
  SEED=$Q/$e.nnue
  echo "[$(date +%H:%M)] $e anti start (epoch-size $TR, from baseline)"
  python3 train.py $B/$e.train.bin --validation-datasets $B/$e.val.bin \
    --validation-size $VR --check-val-every-n-epoch 1 --epoch-size $TR \
    --batch-size 4096 --max-time 00:00:30:00 --max-epochs 100000 \
    --network-save-period 1 --save_top_k 1 \
    --resume-from-model "$SEED" \
    --random-fen-skipping 3 --default-root-dir $R/$e >> $R/$e/train.log 2>&1 &
  TPID=$!
  BEST=-1; BEST_EP=-1
  while kill -0 $TPID 2>/dev/null; do
    sleep 120
    M=$(ls -t $R/$e/lightning_logs/version_*/metrics.csv 2>/dev/null | head -1)
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
        echo "[$(date +%H:%M)] $e: plateau (best $BEST @e$BEST_EP) — stop"
        # graceful drain: SIGTERM, wait for epoch to close, then force
        kill $TPID 2>/dev/null
        for w in 1 2 3 4 5 6; do sleep 10; kill -0 $TPID 2>/dev/null || break; done
        kill -9 $TPID 2>/dev/null
        break
      fi
    fi
    find $R/$e/lightning_logs -name "epoch=*.ckpt" | sort | head -n -3 | xargs -r rm -f
    DU=$(du -sm $R/$e 2>/dev/null | cut -f1)
    if [ "$DU" -gt 12000 ]; then echo "[$(date +%H:%M)] $e: DISK GUARD"; kill $TPID; break; fi
  done
  wait $TPID 2>/dev/null
  # pick newest INTACT ckpt (skip corrupt tail if killed mid-write)
  CK=$(python3 - <<EOF
import glob, re, zipfile
cks = glob.glob("$R/$e/lightning_logs/version_*/checkpoints/epoch=*.ckpt")
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
print("", end="")
EOF
)
  [ -n "$CK" ] && python3 serialize.py "$CK" $R/$e/best.nnue >/dev/null 2>&1 && echo "$e ANTI DONE: best $BEST @e$BEST_EP"
done
echo ANTI-30MIN-QUEUE-DONE
