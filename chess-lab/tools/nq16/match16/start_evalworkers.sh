#!/bin/bash
cd /srv/workspace/chess-active/matches
pgrep -f match_eval_worker.py | xargs -r kill 2>/dev/null
sleep 1
for i in 0 1 2; do nohup python3 match_eval_worker.py $i 3 >> evalworker.log 2>&1 < /dev/null & done
sleep 1
echo workers started
