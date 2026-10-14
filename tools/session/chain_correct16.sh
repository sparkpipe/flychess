#!/bin/bash
# Chain: wait for the 16 puzzle-relabel shards to complete -> export the
# corrected 16-expert bins -> run the verification gate -> if PASS, launch
# the corrected training queue. Any gate failure stops the chain and
# leaves the report for review.
cd /srv/workspace/chess-active
echo "[$(date +%H:%M)] waiting for puzzle relabel (16 shards)..."
while true; do
  D=$(grep -l "SHARD COMPLETE" pz_wk*.log 2>/dev/null | wc -l)
  T=$(ps aux | grep -c "[d]epth_worker.py puzzle-shards")
  echo "[$(date +%H:%M)] shards done: $D/16, workers: $T"
  [ "$D" -ge 16 ] && break
  sleep 300
done
echo "[$(date +%H:%M)] relabel complete — exporting corrected bins"
rm -rf bins-correct16
python3 db_export_correct16.py > export_correct16.log 2>&1
tail -3 export_correct16.log
echo "[$(date +%H:%M)] running verification gate"
python3 verify_correct16.py > verify_correct16.log 2>&1
RC=$?
tail -6 verify_correct16.log
if [ $RC -ne 0 ]; then
  echo "[$(date +%H:%M)] GATE FAILED — training NOT launched"
  exit 1
fi
echo "[$(date +%H:%M)] gate passed — launching corrected training queue"
bash correct16.sh > correct16.log 2>&1
echo "[$(date +%H:%M)] CHAIN COMPLETE"
