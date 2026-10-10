#!/bin/bash
# MoF layer-0 SECTION swarm: dedicated fly per section pool, 3 slots.
cd /home/spec/chess-lab
mkdir -p flies_mof logs_mof
launch() {
  (
    touch "flies_mof/l0_$1.RUNNING"
    POOL="DEGM_$1" python3 tools/mof_l0.py > "logs_mof/$1.log" 2>&1
    rm -f "flies_mof/l0_$1.RUNNING"
    if grep -q "L0-$1-PASSED" "logs_mof/$1.log"; then
      touch "flies_mof/l0_$1.PASS"
    fi
  ) &
}
for pool in $(ls tbpools/DEGM_Ch*_s*.jsonl | xargs -n1 basename | sed "s/DEGM_//;s/.jsonl//"); do
  [ -f "flies_mof/l0_${pool}.PASS" ] && continue
  [ -f "flies_mof/l0_${pool}.RUNNING" ] && continue
  while [ "$(ls flies_mof/*.RUNNING 2>/dev/null | wc -l)" -ge 3 ]; do
    sleep 10
  done
  launch "$pool"
  echo "launched $pool"
done
wait
echo "MOF-SECTION-SWARM-COMPLETE"
