#!/bin/bash
# SELF-PLAY RL CURRICULUM LOOP: runs selfplay_rl.py repeatedly,
# advancing TC on weight convergence. Stops at 60s TC.
R=/mnt/cold-raid6/chess-audit
SP=$R/selfplay_rl
mkdir -p $SP

while true; do
    python3 /srv/workspace/flychess/src/chess-lab/tools/selfplay_rl.py >> $SP/rl_loop.log 2>&1
    if [ -f $SP/RL_DONE ]; then
        echo "CURRICULUM COMPLETE" >> $SP/rl_loop.log
        break
    fi
    sleep 5
done
