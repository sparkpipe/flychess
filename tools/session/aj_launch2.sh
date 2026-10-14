#!/bin/bash
# AJ d10 labeling: N workers (fixed count), nice'd so training wins the CPU.
# Workers claim shards by atomic rename; killing a subshell stops its worker
# permanently (no respawn loop).
cd /srv/workspace/chess-active
NW=${1:-12}
for i in $(seq 1 $NW); do
  (
    while true; do
      S=$(ls aj-kept/aj.???.pgn 2>/dev/null | head -1)
      [ -z "$S" ] && break
      C="$S.claim$i"
      mv "$S" "$C" 2>/dev/null || continue
      nice -n 15 python3 aj_worker.py "$C" "aj-kept/out/$(basename $S .pgn).jsonl" \
        >> aj-kept/wk$i.log 2>&1
      rm -f "$C"
    done
  ) &
  echo "worker $i launched"
done
wait
echo ALL-SHARDS-DONE
