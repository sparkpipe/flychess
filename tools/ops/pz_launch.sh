#!/bin/bash
: "${CHESS_DATA:?CHESS_DATA not set (data root)}"
cd $CHESS_DATA
for i in $(seq 0 15); do
  P=$(printf "pz.%03d" $i)
  [ -f "puzzle-shards/$P.shard" ] || continue
  NODE=pzrelabel-$i SVC=http://127.0.0.1:8123 setsid nohup python3 pd-depthexp/depth_worker.py puzzle-shards/$P.shard > pz_wk$i.log 2>&1 < /dev/null &
done
echo "16 relabel workers launched"
