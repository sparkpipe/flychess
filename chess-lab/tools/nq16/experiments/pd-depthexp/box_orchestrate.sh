#!/bin/bash
# pd-depthexp orchestrator:
#  phase 1 - when the d20 sweep drains, launch workers for shards 6-9
#  phase 2 - when d1 coverage >= 99.5% and GPU is free: build bins, run 12 trainings
E=/srv/workspace/chess-active/pd-depthexp
cd $E
# phase 1: wait for d20 worker to exit, then second wave
(
  while pgrep -f "dbnet_worker2" > /dev/null; do sleep 120; done
  for i in 6 7 8 9; do
    NODE=pddepth-box$i SVC=http://127.0.0.1:8123 nohup python3 depth_worker.py pd.$(printf %02d $i).shard > wk_box$i.log 2>&1 &
  done
  echo "$(date +%H:%M) second wave launched (shards 6-9)"
) > wave2.log 2>&1 &
# phase 2: coverage + GPU gate
(
  while true; do
    sleep 300
    C=$(python3 - << "PYE"
import sqlite3
con = sqlite3.connect("file:/srv/workspace/chess-active/store/games.db?mode=ro", uri=True)
fens = [l.split("\t")[0] for l in open("/srv/workspace/chess-active/pd-depthexp/piece_down.fens")]
have = {f for (f,) in con.execute("select fen from labels where engine='sf17' and depth=1")}
print(f"{sum(1 for f in fens if f in have)/len(fens):.4f}")
PYE
)
    OK=$(python3 -c "print(1 if float('$C') >= 0.995 else 0)")
    GPUFREE=$(pgrep -f "train.py" > /dev/null; echo $?)
    if [ "$OK" = "1" ] && [ "$GPUFREE" = "1" ]; then
      echo "$(date +%H:%M) coverage=$C gpu free - building bins"
      python3 build_bins.py > build_bins.log 2>&1 || { echo "bin build FAILED"; exit 1; }
      echo "$(date +%H:%M) bins built - starting 12 trainings"
      bash train_depths.sh > train_depths.log 2>&1
      echo "$(date +%H:%M) ALL TRAININGS DONE"
      break
    fi
  done
) > pipeline.log 2>&1 &
echo "orchestrator armed"
