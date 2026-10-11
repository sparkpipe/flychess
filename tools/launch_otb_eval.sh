#!/bin/bash
# Launch OTB evaluation workers on this spark node
cd ~/extnvme/phase-moe
for W in $(seq 0 5); do
    WORKER_ID=$W NUM_WORKERS=6 DEPTH=12 \
      INPUT=~/extnvme/phase-moe/otb/shard.txt \
      OUTPUT=~/extnvme/phase-moe/otb/evals_w${W}.txt \
      setsid nohup nice -n 10 python3 otb_eval_fleet.py \
      >> otb/worker_${W}.log 2>&1 < /dev/null &
done
sleep 2
echo "$(hostname -s): $(pgrep -c -f otb_eval) eval workers running"
