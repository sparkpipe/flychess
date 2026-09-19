#!/bin/bash
# MoF layer-0 swarm: 15 dedicated chapter flies, 3 parallel slots.
cd /home/spec/chess-lab
mkdir -p flies_mof logs_mof
for ch in $(seq 1 15); do
  while [ "$(jobs -r | wc -l)" -ge 3 ]; do sleep 5; done
  if [ -f "flies_mof/l0_ch${ch}.PASS" ]; then
    echo "ch${ch} already PASSED (skip)"
    continue
  fi
  (
    CH=$ch python3 tools/mof_l0.py > logs_mof/ch${ch}.log 2>&1
    if grep -q "L0-CH${ch}-PASSED" logs_mof/ch${ch}.log; then
      touch "flies_mof/l0_ch${ch}.PASS"
    fi
  ) &
done
wait
echo "MOF-L0-SWARM-COMPLETE"
