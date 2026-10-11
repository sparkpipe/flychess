#!/bin/bash
# one instance each: stage A (50-ply fens), miniatures, then stage B
# chained after stage A; reader already trained
cd /home/spec/chess-lab
rm -f gambit/fens*.txt gambit/evals*.txt
setsid nohup python3 tools/gambit_stageA.py > gambit_a.out 2>&1 < /dev/null &
setsid nohup python3 tools/miniature_pools.py > mini.out 2>&1 < /dev/null &
( while ! grep -q "STAGE-A-COMPLETE" gambit_a.out 2>/dev/null; do sleep 60; done
  python3 tools/gambit_stageB.py > gambit_b.out 2>&1 ) &
echo "launched: stage A + miniatures; stage B chained"
