#!/bin/bash
# Complete OTB dump: parallel using line-based split
set -u
PGN=/home/spec/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn
OUTDIR=/home/spec/chess-lab/otb_dump_shards
NPROC=12
mkdir -p $OUTDIR

# Kill any existing dump
pkill -f otb_dump 2>/dev/null; sleep 1

echo "Splitting PGN by lines into ${NPROC} shards..."
split -n l/${NPROC} -d "$PGN" "$OUTDIR/shard_"
echo "Split done: $(ls $OUTDIR/shard_* | wc -l) shards"

echo "Processing ${NPROC} shards in parallel..."
for I in $(seq -w 0 $((NPROC - 1))); do
    (
    python3 -c "
import sys, os
sys.path.insert(0, '/home/spec/chess-lab/tools')
os.environ['PGN'] = '$OUTDIR/shard_$I'
os.environ['OUT'] = '$OUTDIR/positions_$I.txt'
exec(open('/home/spec/chess-lab/tools/otb_dump.py').read())
" > "$OUTDIR/log_$I.txt" 2>&1 &
    ) &
done

echo "Waiting for all shards..."
wait

# Combine results
cat $OUTDIR/positions_*.txt > /home/spec/chess-lab/otb_full_dump.txt
TOTAL=$(wc -l < /home/spec/chess-lab/otb_full_dump.txt)
echo "FULL OTB DUMP COMPLETE: ${TOTAL} positions"
echo "OTB-DUMP-FULL-COMPLETE"
