#!/bin/bash
# Relaunch AJ d10 labeling after the hardware-update suspension.
# 1) Rename all .claimN files back to plain .pgn so workers re-pick them
#    (aj_worker_resume.py continues each partial out/*.jsonl).
# 2) N nice'd workers, claim-by-rename, same as aj_launch2.sh.
# Usage: aj_resume_launch.sh [NW]
OPS="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
DATA=${DATA_DIR:-/mnt/flychess/chess-active}
cd "$DATA"
NW=${1:-13}
for C in aj-kept/aj.*.pgn.claim*; do
  [ -e "$C" ] || continue
  mv "$C" "${C%.claim*}"
done
echo "claims restored: $(ls aj-kept/aj.???.pgn 2>/dev/null | wc -l) shards available"
for i in $(seq 1 $NW); do
  (
    while true; do
      S=$(ls aj-kept/aj.???.pgn 2>/dev/null | head -1)
      [ -z "$S" ] && break
      C="$S.claim$i"
      mv "$S" "$C" 2>/dev/null || continue
      nice -n 15 python3 "$OPS/aj_worker_resume.py" "$C" "aj-kept/out/$(basename $S .pgn).jsonl" \
        >> aj-kept/wk$i.log 2>&1
      rm -f "$C"
    done
  ) &
  echo "worker $i launched"
done
wait
echo ALL-SHARDS-DONE
