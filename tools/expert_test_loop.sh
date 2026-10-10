#!/bin/bash
# Per-expert strength tests as training completes; cumulative results.
# For each finished expert: serialize, 3-game matches at skill 12 and 15,
# final val loss -> cumulative table.
R=/mnt/cold-raid6/chess-audit
RES=$R/expert_strength.txt
ORDER="balanced_l0 balanced_l1 nvb exchanges balanced_l2 oppb bvr dvoretsky balanced_l3 nvr rv2m qvmat tactics tb_training"
mkdir -p "$R/nets"
: > "$RES"
echo "expert  skill12  skill15  val_loss" >> "$RES"

tested() { grep -q "^$1 " "$RES"; }

while true; do
  alldone=0
  [ -f "$R/TRAINING_DONE" ] && alldone=1
  for name in $ORDER; do
    tested "$name" && continue
    ck=$(ls -t "$R/runs/$name"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -z "$ck" ] && continue
    # only test when training has MOVED PAST this expert (next TRAIN line newer)
    if [ "$name" != "tb_training" ]; then
      grep -q "=== TRAIN $name " "$R/train_all.log" || continue
      nxt=$(grep "=== TRAIN " "$R/train_all.log" | sed -n "/=== TRAIN $name /{n;p}")
      [ -z "$nxt" ] && [ "$alldone" = 0 ] && continue
    elif [ "$alldone" = 0 ]; then
      continue
    fi
    # still writing? skip if ckpt modified in the last 3 minutes
    [ -n "$(find "$ck" -mmin -3 2>/dev/null)" ] && continue
    echo "TESTING $name $(date -u)" >> "$RES.testlog"
    cd /home/spec/nnue-pytorch
    python3 serialize.py "$ck" "$R/nets/$name.nnue" >> "$RES.testlog" 2>&1
    vl=$(grep -oE "val_loss=[0-9.]+" "$R/runs/$name.trainlog" | tail -1 | cut -d= -f2)
    s12=$(OUR_NET=$R/nets/$name.nnue OPP_SKILL=12 OPP_DEPTH=8 \
      python3 /home/spec/chess-lab/tools/quick_match.py 2>/dev/null | grep -oE "score=[0-9.]+" | tail -1 | cut -d= -f2)
    s15=$(OUR_NET=$R/nets/$name.nnue OPP_SKILL=15 OPP_DEPTH=10 \
      python3 /home/spec/chess-lab/tools/quick_match.py 2>/dev/null | grep -oE "score=[0-9.]+" | tail -1 | cut -d= -f2)
    echo "$name  ${s12:-NA}  ${s15:-NA}  ${vl:-NA}" >> "$RES"
  done
  [ "$alldone" = 1 ] && { tested tb_training || continue; }
  [ "$alldone" = 1 ] && tested tb_training && break
  sleep 180
done
touch "$R/EXPERT_TESTS_DONE"
echo "=== EXPERT TESTS COMPLETE $(date -u) ===" >> "$RES"
