#!/bin/bash
# Deployed to each spark — launches 8 data-gen workers
cd ~/extnvme/phase-moe
rm -f worker_*.log
for W in $(seq 1 8); do
    WORKER_ID=$W NODE=$(hostname -s) DEPTH=10 TIMEOUT_MIN=60 \
      setsid nohup nice -n 10 python3 gen_fleet.py >> worker_$W.log 2>&1 < /dev/null &
done
sleep 2
echo "$(hostname -s): $(pgrep -c -f 'python3 gen_fleet') workers running"
