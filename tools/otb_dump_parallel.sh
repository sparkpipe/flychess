#!/bin/bash
# Parallel OTB dump: splits the PGN into N byte-range chunks,
# runs dump on each in parallel, concatenates results.
set -u
PGN=/home/spec/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn
OUTDIR=/home/spec/chess-lab/otb_shards
NPROC=12
mkdir -p $OUTDIR

# kill the slow single-threaded dump
pkill -f otb_dump.py 2>/dev/null
sleep 1

# split the PGN into byte ranges
SIZE=$(stat -c %s $PGN)
CHUNK=$((SIZE / NPROC + 1))

echo "splitting ${SIZE} bytes into ${NPROC} chunks of ${CHUNK}..."

for I in $(seq 0 $((NPROC - 1))); do
    START=$((I * CHUNK))
    END=$(( (I + 1) * CHUNK ))
    [ $END -gt $SIZE ] && END=$SIZE

    # extract byte range and snap to game boundary
    dd if=$PGN bs=1 skip=$START count=$((END - START)) 2>/dev/null \
        | head -c $((END - START)) > $OUTDIR/shard_$I.pgn

    # snap to [Event boundary: trim first partial game, trim last partial
    python3 -c "
import sys
data = open('$OUTDIR/shard_$I.pgn', 'rb').read()
# find first game boundary
first = data.find(b'[Event ')
if first > 0:
    data = data[first:]
# find last complete game (ends with blank line after moves)
last = data.rfind(b'\n\n[Event ')
if last > 0:
    data = data[:last + 2]
open('$OUTDIR/shard_$I.pgn', 'wb').write(data)
print(f'shard $I: {len(data)} bytes')
" &
done
wait

echo "launching ${NPROC} parallel dump workers..."
for I in $(seq 0 $((NPROC - 1))); do
    PGN=$OUTDIR/shard_$I.pgn \
    OUT=$OUTDIR/positions_$I.txt \
    nice -n 5 python3 /home/spec/chess-lab/tools/otb_dump.py &
done
wait

# concatenate all shard positions
cat $OUTDIR/positions_*.txt > /home/spec/chess-lab/otb_all_positions.txt
TOTAL=$(wc -l < /home/spec/chess-lab/otb_all_positions.txt)
echo "OTB PARALLEL DUMP COMPLETE: ${TOTAL} positions"
