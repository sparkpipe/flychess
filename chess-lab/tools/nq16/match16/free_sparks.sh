#!/bin/bash
# Stop ALL chess usage on this spark: anti trainer+watcher, d20 worker, sf17 evals,
# 23-era queue workers/trainers. Leaves sparkpipe infra (queue dispatcher, telemetry) alone.
for w in $(pgrep -f "anti_watch_spar[k]\.sh"); do kill $w 2>/dev/null; done
for p in $(pgrep -f "train\.py .*runanti_"); do kill $p 2>/dev/null; done
for d in $(pgrep -f "d20_spark_worke[r]"); do kill $d 2>/dev/null; done
for q in $(pgrep -f "train23_queu[e]"); do kill $q 2>/dev/null; done
for t in $(pgrep -f "train\.py .*train2[3]"); do kill $t 2>/dev/null; done
sleep 3
for p in $(pgrep -f "train\.py .*runanti_"); do kill -9 $p 2>/dev/null; done
for q in $(pgrep -f "train23_queu[e]"); do kill -9 $q 2>/dev/null; done
for s in $(pgrep -x sf17_arm); do kill $s 2>/dev/null; done
for s in $(pgrep -x sf17); do kill $s 2>/dev/null; done
sleep 1
left=$(pgrep -fa "train\.py|runanti|anti_watch|d20_spark|sf17|train23" | grep -v pgrep | grep -v "pgrep -fa" | head -5)
if [ -n "$left" ]; then
  echo "REMAINING on $(hostname):"
  echo "$left"
else
  echo "$(hostname): all chess usage stopped"
fi
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
