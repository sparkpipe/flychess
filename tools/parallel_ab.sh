#!/bin/bash
# Parallel 3-game strength tests: curated net vs self-play net
cd /home/spec/chess-lab

# Test curated net
for SKILL in 8 12 15 18 20; do
    OUR_NET=/home/spec/chess-lab/curated_net.nnue \
      GAMES=3 TPM=0.5 OPP_SKILL=$SKILL \
      python3 tools/quick_match.py > match_curated_s${SKILL}.out 2>&1 &
done

# Test self-play baseline net for direct comparison (same conditions)
for SKILL in 15 18 20; do
    OUR_NET=/home/spec/chess-lab/our_net_v2.nnue \
      GAMES=3 TPM=0.5 OPP_SKILL=$SKILL \
      python3 tools/quick_match.py > match_baseline_s${SKILL}.out 2>&1 &
done

wait
echo "=== ALL A/B TESTS DONE ==="
echo ""
echo "CURATED (OTB master games, trajectory-filtered):"
for SKILL in 8 12 15 18 20; do
    echo "  skill $SKILL: $(grep FINAL match_curated_s${SKILL}.out 2>/dev/null)"
done
echo ""
echo "BASELINE (self-play, same training budget):"
for SKILL in 15 18 20; do
    echo "  skill $SKILL: $(grep FINAL match_baseline_s${SKILL}.out 2>/dev/null)"
done
