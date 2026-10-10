#!/bin/bash
# Fleet-wide NNUE data generation (operator: "proceed" 2026-09-27)
# Stages on each spark: local NVMe dir, SF binary (ARM build), gen script.
# Runs N_WORKERS parallel data-gen processes per node, each writing
# timestamped .bin chunks. Chunks collect to rtx5090 when done.
# Usage: bash fleet_datagen.sh [minutes]  (default 60 = ~1B positions fleet-wide)
set -u

MINUTES=${1:-60}
NODES="spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 sparka sparkb sparkc sparkd sparke sparkf"
# spark9 = controller standby; adjust if topology differs
WORKERS_PER_NODE=8
GAMES_PER_CHUNK=500
DEPTH=${DEPTH:-10}
BASE=/home/spark2/extnvme/phase-moe   # template from spark2 pilot
RMT=/home/spec/chess-lab/fleet_data

echo "=== fleet datagen: ${MINUTES}min, ${#NODES} nodes, ${WORKERS_PER_NODE} workers/node ==="

# Phase 1: stage all nodes (sequential — fast, just scp + mkdir)
for NODE in $NODES; do
    echo "--- staging $NODE..."
    ssh -o ConnectTimeout=5 $NODE "
        mkdir -p /home/spark2/extnvme/phase-moe/data
        # copy SF binary from spark2 (already built for ARM)
        cp /home/spark2/extnvme/phase-moe/sf/src/stockfish /home/spark2/extnvme/phase-moe/sf_arm 2>/dev/null || true
        # copy gen script from spark2
        cp /home/spark2/extnvme/phase-moe/gen_smoke_data.py /home/spark2/extnvme/phase-moe/ 2>/dev/null || true
    " 2>/dev/null || echo "WARN: $NODE unreachable, skipping"
done

# Phase 2: launch workers on all nodes
for NODE in $NODES; do
    echo "--- launching $NODE..."
    ssh $NODE "
        cd /home/spark2/extnvme/phase-moe
        for W in \$(seq 1 $WORKERS_PER_NODE); do
            SEED=\$((W + RANDOM))
            OUTFILE=data/chunk_\${NODE}_w\${W}_\$(date +%s).bin
            GAMES=$GAMES_PER_CHUNK DEPTH=$DEPTH SEED=\$SEED \
              nohup nice -n 10 python3 gen_fleet.py > /dev/null 2>&1 &
        done
        echo \"  \$WORKERS_PER_NODE workers launched\"
    " 2>/dev/null || echo "WARN: $NODE launch failed"
done

echo "=== all workers running; collecting after ${MINUTES}min ==="
sleep $((MINUTES * 60))

# Phase 3: collect to rtx5090
mkdir -p $RMT
TOTAL_ROWS=0
for NODE in $NODES; do
    COUNT=\$(ssh $NODE "ls /home/spark2/extnvme/phase-moe/data/*.bin 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    SIZE=\$(ssh $NODE "du -sb /home/spark2/extnvme/phase-moe/data/ 2>/dev/null | cut -f1" 2>/dev/null || echo 0)
    ROWS=\$((SIZE / 40))
    TOTAL_ROWS=\$((TOTAL_ROWS + ROWS))
    echo "  $NODE: $COUNT chunks, $ROWS rows"
    # rsync chunks to controller
    rsync -a --remove-source-files $NODE:/home/spark2/extnvme/phase-moe/data/ $RMT/ 2>/dev/null || \
        scp $NODE:/home/spark2/extnvme/phase-moe/data/*.bin $RMT/ 2>/dev/null || true
done

echo "=== COMPLETE: $TOTAL_ROWS total positions across fleet ==="
echo "Data in: $RMT"
