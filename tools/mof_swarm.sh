#!/bin/bash
# MoF layer-0 swarm: 15 dedicated chapter flies, 3 parallel slots.
# Slot counting via RUNNING markers (jobs(1) is unreliable here).
cd /home/spec/chess-lab
mkdir -p flies_mof logs_mof
launch() {
  (
    touch "flies_mof/l0_ch$1.RUNNING"
    CH=$1 python3 tools/mof_l0.py > "logs_mof/ch$1.log" 2>&1
    rm -f "flies_mof/l0_ch$1.RUNNING"
    if grep -q "L0-CH$1-PASSED" "logs_mof/ch$1.log"; then
      touch "flies_mof/l0_ch$1.PASS"
    fi
  ) &
}
for ch in $(seq 1 15); do
  [ -f "flies_mof/l0_ch${ch}.PASS" ] && continue
  [ -f "flies_mof/l0_ch${ch}.RUNNING" ] && continue
  while [ "$(ls flies_mof/*.RUNNING 2>/dev/null | wc -l)" -ge 3 ]; do
    sleep 10
  done
  launch "$ch"
  echo "launched ch${ch}"
done
wait
echo "MOF-L0-SWARM-COMPLETE"
