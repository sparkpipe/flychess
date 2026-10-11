#!/bin/bash
# 12 fresh nets, one per eval depth, on the piece_down set. Small budget per
# depth (10 min): we compare val_corr across depths, not chase max strength.
set -u
cd /srv/workspace/flychess/src/nnue-pytorch
B=/srv/workspace/chess-active/pd-depthexp/bins
R=/srv/workspace/chess-active/pd-depthexp/runs
mkdir -p $R
for D in 1 2 3 4 5 6 7 8 9 10 11 12; do
  [ -f $R/d$D/best.nnue ] && continue
  TR=$(( $(stat -c %s $B/d$D.train.bin) / 40 ))
  VR=$(( $(stat -c %s $B/d$D.val.bin) / 40 )); [ $VR -gt 20000 ] && VR=20000
  RUN=$R/d$D; mkdir -p $RUN
  echo "[$(date +%H:%M)] d$D: $TR recs"
  timeout 700 python3 train.py $B/d$D.train.bin --validation-datasets $B/d$D.val.bin \
    --validation-size $VR --check-val-every-n-epoch 1 --epoch-size 500000 \
    --batch-size 4096 --max-epochs 100000 --network-save-period 1 --save_top_k 1 \
    --random-fen-skipping 3 --default-root-dir $RUN >> $RUN/train.log 2>&1
  python3 - "$RUN" << "PYE"
import csv, glob, re, subprocess, sys
R = sys.argv[1]
best = (-2, -1)
for m in glob.glob(f"{R}/lightning_logs/version_*/metrics.csv"):
    for r in csv.DictReader(open(m)):
        c = r.get("val_corr")
        if c not in (None, ""):
            try: v = float(c)
            except ValueError: continue
            if v > best[0]: best = (v, int(r.get("epoch", -1)))
cks = sorted(glob.glob(f"{R}/lightning_logs/version_*/checkpoints/epoch=*.ckpt"),
             key=lambda f: int(re.search(r"epoch=(\d+)", f).group(1)) if re.search(r"epoch=(\d+)", f) else -1)
if cks and best[0] > -2:
    subprocess.run(["python3", "serialize.py", cks[-1], f"{R}/best.nnue"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"d-depth RESULT corr={best[0]:.4f} @e{best[1]}")
PYE
done
echo "=== SUMMARY val_corr by eval depth ==="
for D in 1 2 3 4 5 6 7 8 9 10 11 12; do
  grep -h "RESULT corr" $R/d$D/train.log 2>/dev/null | tail -1 | sed "s/^/d$D /"
done
