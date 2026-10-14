#!/bin/bash
# Stop all d20 workers on all sparks (operator: sparks offline for chess).
# Shards are preserved (data already produced stays).
for s in spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 $s "
    pgrep -f start_worker3 | xargs -r kill 2>/dev/null
    pgrep -f dbnet_worker3.py | xargs -r kill 2>/dev/null
    sleep 2
    pgrep -f dbnet_worker3.py | xargs -r kill -9 2>/dev/null
    pgrep -x sf17_arm | xargs -r kill 2>/dev/null
    echo stopped" 2>/dev/null && echo "$s: stopped" || echo "$s: unreachable"
done
