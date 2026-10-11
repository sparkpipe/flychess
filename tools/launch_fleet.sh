#!/bin/bash
# Simplified fleet launcher: stages SF + gen script on all sparks,
# launches N workers per node, each writing local NVMe chunks.
# Usage: bash launch_fleet.sh [minutes]
set -u
MINUTES=${1:-60}
NODES="spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 sparka sparkb sparkc sparkd sparke sparkf"
WORKERS=8
DEPTH=10
STAGE=/home/spark2/extnvme/phase-moe

echo "=== staging ${#NODES} nodes ==="
for NODE in $NODES; do
    # stage: copy from spark2 (fast local NVMe → NVMe)
    ssh -o ConnectTimeout=5 $NODE "
        mkdir -p $STAGE/data
        # SF binary and gen script — pull from spark2 via controller relay
        # (sparks can't ssh each other directly)
        true
    " 2>/dev/null || echo "WARN: $NODE staging failed"
done

# relay binaries via controller (rtx5090)
echo "=== relaying binaries via controller ==="
for NODE in $NODES; do
    [ "$NODE" = "spark2" ] && continue   # spark2 already has everything
    ssh $NODE "mkdir -p $STAGE/sf/src $STAGE" 2>/dev/null
    # spark2 → controller → spark (two-hop relay)
    ssh spark2 "cat $STAGE/sf/src/stockfish" | ssh $NODE "cat > $STAGE/sf/src/stockfish; chmod +x $STAGE/sf/src/stockfish"
    scp -q /home/spec/chess-lab/tools/gen_fleet.py $NODE:$STAGE/
    echo "  $NODE staged"
done

echo "=== launching ${WORKERS} workers per node ==="
for NODE in $NODES; do
    ssh $NODE "
        cd $STAGE
        for W in \$(seq 1 $WORKERS); do
            WORKER_ID=\$W NODE=$NODE DEPTH=$DEPTH TIMEOUT_MIN=$MINUTES \
              setsid nohup nice -n 10 python3 gen_fleet.py \
              >> worker_\$W.log 2>&1 < /dev/null &
        done
        echo '  launched'
    " 2>/dev/null
done

echo "=== fleet running for ${MINUTES} minutes ==="
echo "monitor: ssh any-spark 'ls $STAGE/data | wc -l; du -sh $STAGE/data'"
