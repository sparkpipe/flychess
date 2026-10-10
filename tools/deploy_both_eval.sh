#!/bin/bash
# Deploy the BOTH-SIDES dump depth-12 evaluation across all 14 sparks.
# Shards live on the RAID; per-node workers write to ~/extnvme/phase-moe/otb2/.
set -u
SRC=/mnt/cold-raid6/chess-audit/otb_complete_dump_both.txt
STAGE=/mnt/cold-raid6/chess-audit/both_shards
NODES="spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 sparka sparkb sparkc sparkd sparke sparkf"

mkdir -p "$STAGE"
if [ ! -f "$STAGE/shard_00" ] || [ "$(wc -l < "$STAGE/shard_00")" -lt 1000 ]; then
  rm -f "$STAGE"/shard_* 2>/dev/null
  echo "splitting $(wc -l < "$SRC") lines into 14 shards"
  split -n l/14 -d -a 2 "$SRC" "$STAGE/shard_"
fi

i=0
for N in $NODES; do
  id=$(printf "%02d" "$i"); i=$((i+1))
  if ! timeout 8 ssh -o ConnectTimeout=6 "$N" true 2>/dev/null; then
    echo "$N UNREACHABLE"; continue
  fi
  ssh "$N" "mkdir -p ~/extnvme/phase-moe/otb2 && pkill -f \"otb_eval_[f]leet\" 2>/dev/null; sleep 1; rm -f ~/extnvme/phase-moe/otb2/evals_w*.txt ~/extnvme/phase-moe/otb2/worker_*.log; true"
  scp -q "$STAGE/shard_$id" "$N:~/extnvme/phase-moe/otb2/shard.txt" || { echo "$N scp FAILED"; continue; }
  scp -q /home/spec/chess-lab/tools/otb_eval_fleet.py "$N:~/extnvme/phase-moe/" || { echo "$N script FAILED"; continue; }
  ssh "$N" "cd ~/extnvme/phase-moe/otb2 && for W in \$(seq 0 5); do WORKER_ID=\$W NUM_WORKERS=6 DEPTH=12 INPUT=\$HOME/extnvme/phase-moe/otb2/shard.txt OUTPUT=\$HOME/extnvme/phase-moe/otb2/evals_w\$W.txt setsid nohup nice -n 10 python3 \$HOME/extnvme/phase-moe/otb_eval_fleet.py >> worker_\$W.log 2>&1 < /dev/null & done; sleep 3; echo \"\$HOSTNAME: \$(pgrep -c -f otb_eval_fleet) workers\""
done
echo DEPLOY_BOTH_DONE
