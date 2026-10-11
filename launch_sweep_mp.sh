#!/bin/bash
# launch N mod-partitioned sweep workers (multiprocess parallelism)
cd /home/spec/chess-lab
N=${1:-4}
for k in $(seq 0 $((N-1))); do
  SWEEP_MOD=$N SWEEP_REM=$k W=1 GATE_CH=128 nohup python3 tools/sweep_singles.py >> sweep_mp_$k.out 2>&1 < /dev/null &
done
echo "$N workers launched"
