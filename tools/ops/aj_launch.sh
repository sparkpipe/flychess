#!/bin/bash
: "${CHESS_DATA:?CHESS_DATA not set (data root)}"
cd $CHESS_DATA
for i in $(seq 1 16); do
  (
    while true; do
      S=$(ls aj-kept/aj.???.pgn 2>/dev/null | head -1)
      [ -z "$S" ] && break
      C="$S.claim$i"
      mv "$S" "$C" 2>/dev/null || continue
      python3 aj_worker.py "$C" "aj-kept/out/$(basename $S .pgn).jsonl" \
        >> aj-kept/wk$i.log 2>&1
      rm -f "$C"
    done
  ) &
done
wait
echo ALL-SHARDS-DONE
