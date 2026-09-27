#!/bin/bash
# Parallel 3-game strength tests across skill levels
cd /home/spec/chess-lab
for SKILL in 8 12 15 18 20; do
    GAMES=3 TPM=0.5 OPP_SKILL=$SKILL \
      python3 tools/quick_match.py > match_s${SKILL}.out 2>&1 &
done
wait
echo "=== ALL PARALLEL TESTS DONE ==="
for SKILL in 8 12 15 18 20; do
    echo "skill $SKILL: $(grep FINAL match_s${SKILL}.out)"
done
