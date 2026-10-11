#!/bin/bash
# Post-eval chain: wait for all 84 fleet workers to finish, then
# collect -> extract -> census -> pack the audited per-expert bins.
# Marker-file based (no pgrep patterns). Runs detached on rtx5090.
set -u
R=/mnt/cold-raid6/chess-audit
NODES="spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 sparka sparkb sparkc sparkd sparke sparkf"
LOG="$R/chain.log"
exec >> "$LOG" 2>&1
echo "=== chain armed $(date -u) ==="

# 1. wait for completion: every node 6x DONE in worker logs
while true; do
  done_all=1
  for s in $NODES; do
    n=$(ssh -o ConnectTimeout=8 "$s" \
        "grep -h DONE ~/extnvme/phase-moe/otb/worker_*.log 2>/dev/null | wc -l" \
        2>/dev/null)
    [ "${n:-0}" -ge 6 ] || { done_all=0; break; }
  done
  [ "$done_all" = 1 ] && break
  sleep 120
done
echo "=== all workers DONE $(date -u) ==="

# 2. stream-collect in node order (overwrites the junk partial file)
rm -f "$R/otb_full_evals.txt"
for s in $NODES; do
  ssh -o ConnectTimeout=8 "$s" \
    "cat ~/extnvme/phase-moe/otb/evals_w0.txt ~/extnvme/phase-moe/otb/evals_w1.txt \
        ~/extnvme/phase-moe/otb/evals_w2.txt ~/extnvme/phase-moe/otb/evals_w3.txt \
        ~/extnvme/phase-moe/otb/evals_w4.txt ~/extnvme/phase-moe/otb/evals_w5.txt" \
    >> "$R/otb_full_evals.txt" || echo "COLLECT_FAILED $s"
done
LINES=$(wc -l < "$R/otb_full_evals.txt")
echo "collected $LINES evals (target 63318852, slack for eval failures)"
if [ "$LINES" -lt 62000000 ]; then
  echo "FATAL: collection short — aborting before extract"
  touch "$R/CHAIN_FAILED"
  exit 1
fi

# 3. extract segments (v2)
cd /home/spec/chess-lab
python3 tools/segment_extractor_v2.py < "$R/otb_full_evals.txt" \
  > "$R/segments_v2_full.jsonl" || { touch "$R/CHAIN_FAILED"; exit 1; }
echo "extraction done $(date -u)"

# 4. census + pack the audited bins
rm -rf "$R/expert_bins"
python3 tools/expert_census.py "$R/segments_v2_full.jsonl" > "$R/census_full.txt"
python3 tools/pack_expert_bins.py "$R/segments_v2_full.jsonl" "$R/expert_bins" \
  > "$R/pack_full.txt" || { touch "$R/CHAIN_FAILED"; exit 1; }

touch "$R/CHAIN_COMPLETE"
echo "=== CHAIN COMPLETE $(date -u) ==="
