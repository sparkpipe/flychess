#!/bin/bash
# Both-sides post-chain: collect the 125.9M evals, extract, census, pack
# the REBUILT per-expert bins (winner + loser positions, four-category
# selection). Marker-based. Also assembles tactics.bin eval inputs.
set -u
R=/mnt/cold-raid6/chess-audit
NODES="spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 sparka sparkb sparkc sparkd sparke sparkf"
LOG="$R/chain_both.log"
exec >> "$LOG" 2>&1
echo "=== both-chain armed $(date -u) ==="

rm -f "$R/otb_both_evals.txt"
for s in $NODES; do
  ssh -o ConnectTimeout=8 "$s" \
    "cat ~/extnvme/phase-moe/otb2/evals_w0.txt ~/extnvme/phase-moe/otb2/evals_w1.txt \
        ~/extnvme/phase-moe/otb2/evals_w2.txt ~/extnvme/phase-moe/otb2/evals_w3.txt \
        ~/extnvme/phase-moe/otb2/evals_w4.txt ~/extnvme/phase-moe/otb2/evals_w5.txt" \
    >> "$R/otb_both_evals.txt" || echo "COLLECT_FAILED $s"
done
L=$(wc -l < "$R/otb_both_evals.txt")
echo "collected $L evals (target 125872986)"
[ "$L" -lt 123000000 ] && { echo FATAL_SHORT; touch "$R/BOTH_FAILED"; exit 1; }

cd /home/spec/chess-lab
python3 tools/segment_extractor_v2.py < "$R/otb_both_evals.txt" \
  > "$R/segments_v2_both.jsonl" || { touch "$R/BOTH_FAILED"; exit 1; }
echo "extraction done $(date -u)"

python3 tools/expert_census.py "$R/segments_v2_both.jsonl" > "$R/census_both.txt"
rm -rf "$R/expert_bins_both"
python3 tools/pack_expert_bins.py "$R/segments_v2_both.jsonl" "$R/expert_bins_both" \
  > "$R/pack_both.txt" || { touch "$R/BOTH_FAILED"; exit 1; }

touch "$R/BOTH_COMPLETE"
echo "=== BOTH-CHAIN COMPLETE $(date -u) ==="
