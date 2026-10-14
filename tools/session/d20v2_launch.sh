#!/bin/bash
# Deploy 16-core d20 workers on ALL 16 sparks (including spark5).
# Cursor protocol: GET /next -> search -> POST /result. No files.
cd /srv/workspace/chess-active
for s in spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf; do
  # kill any existing workers on this spark
  ssh -o BatchMode=yes -o ConnectTimeout=8 $s "pkill -f d20_[w]orker 2>/dev/null; pkill -f sf17_[a]rm 2>/dev/null; true" 2>/dev/null
done
echo "old workers killed"
sleep 2
for s in spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 $s "
    setsid nohup python3 -u ~/d20_worker.py $s 16 http://192.168.50.4:8125 ~/sf17_arm >> ~/d20full.log 2>&1 < /dev/null &
    sleep 1
    echo \"$s: \$(pgrep -c sf17_arm) SF procs\"
  " 2>/dev/null
done
echo "ALL DEPLOYED"
