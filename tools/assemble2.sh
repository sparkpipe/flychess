#!/bin/bash
# Corrected final assembly: raw-UCI resumable aug evals -> aug pack -> marker.
set -u
R=/mnt/cold-raid6/chess-audit
LOG="$R/assemble2.log"
exec >> "$LOG" 2>&1
echo "=== assemble2 armed $(date -u) ==="
cd /home/spec/chess-lab

for W in 0 1 2 3 4 5 6 7; do
  INPUT=$R/gambit_aug_positions.txt OUTPUT=$R/gaug_evals/w$W.txt \
    WORKER_ID=$W NUM_WORKERS=8 DEPTH=12 \
    setsid nohup nice -n 8 python3 tools/eval_raw.py \
    > "$R/gaug_evals/raw_w$W.log" 2>&1 < /dev/null &
done
while true; do
  d=$(grep -lh "DONE" "$R"/gaug_evals/raw_w*.log 2>/dev/null | wc -l)
  [ "$d" -ge 8 ] && break
  sleep 120
done
echo "aug evals done $(date -u)"

python3 tools/pack_aug.py "$R/gaug_evals" "$R/expert_bins_both" \
  > "$R/aug_pack.txt" 2>&1 || { touch "$R/ASSEMBLE_FAILED"; exit 1; }
tail -20 "$R/aug_pack.txt"
touch "$R/ASSEMBLE_COMPLETE"
echo "=== ASSEMBLE2 COMPLETE $(date -u) ==="
