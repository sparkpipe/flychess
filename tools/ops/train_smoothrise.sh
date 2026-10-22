#!/bin/bash
# Smooth-rise net — operator recipe (2026-10-18): train to MAX (plateau)
# on d12 labels, then warm-finish on d20 labels. Two phases, plateau =
# 15 stale validations, 12h safety cap per phase.
: "${BIN_DIR:?BIN_DIR not set (holds d12/ and d20/ bins)}"
: "${RUN_DIR:?RUN_DIR not set (checkpoints + best.nnue land here)}"
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO/nnue-pytorch"
mkdir -p "$RUN_DIR"

run_phase() {  # $1=depth $2=resume-ckpt(«"")
  local D=$1 RESUME=$2
  local SR="$RUN_DIR/stage_d$D"
  mkdir -p "$SR"
  local TR=$(( $(stat -c %s "$BIN_DIR/d$D/train.bin") / 40 ))
  local VR=$(( $(stat -c %s "$BIN_DIR/d$D/val.bin") / 40 ))
  [ $VR -gt 50000 ] && VR=50000
  local R=""
  [ -n "$RESUME" ] && R="--resume-from-checkpoint $RESUME"
  echo "[$(date +%H:%M)] phase d$D: $TR recs, plateau-stop (15 stale), resume=${RESUME:+yes}"
  python3 train.py "$BIN_DIR/d$D/train.bin" \
    --validation-datasets "$BIN_DIR/d$D/val.bin" \
    --validation-size $VR --check-val-every-n-epoch 1 --epoch-size $TR \
    --batch-size 4096 --max-time 12:00:00:00 --max-epochs 100000 \
    --network-save-period 1 --save_top_k 1 \
    $R \
    --random-fen-skipping 3 --default-root-dir "$SR" >> "$SR/train.log" 2>&1 &
  local TPID=$!
  local BEST=-1 BEST_EP=-1
  while kill -0 $TPID 2>/dev/null; do
    sleep 120
    local M=$(ls -t "$SR"/lightning_logs/version_*/metrics.csv 2>/dev/null | head -1)
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
      local GAP=$(( L2 - BEST_EP ))
      if [ $GAP -ge 15 ]; then
        echo "[$(date +%H:%M)] d$D: plateau (best $BEST @e$BEST_EP, gap=$GAP) — advancing"
        kill $TPID; sleep 10; kill -9 $TPID 2>/dev/null; break
      fi
    fi
  done
  wait $TPID 2>/dev/null
  BEST_CK=$(python3 << EOF
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
  [ -z "$BEST_CK" ] && { echo "phase d$D FAILED — no ckpt"; exit 1; }
  echo "  d$D done: best corr $BEST @e$BEST_EP"
}

run_phase 12 ""
run_phase 20 "$BEST_CK"
python3 serialize.py "$BEST_CK" "$RUN_DIR/best.nnue" >/dev/null 2>&1
echo "[$(date +%H:%M)] smoothrise net DONE -> $RUN_DIR/best.nnue"
