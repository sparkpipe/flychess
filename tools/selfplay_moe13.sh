#!/bin/bash
# SELF-PLAY: moe13 (hard-routed, current best nets) plays against itself,
# generating training data for the stacked head. Games at 1s/move on CPU.
# Both sides use the same engine+loadout; positions+outcomes+evals logged.
set -u
R=/mnt/cold-raid6/chess-audit
A=$HOME/arena
OUT=$R/selfplay
mkdir -p "$OUT/games"
cd "$A"

OPTS="option.EvalFile=$R/nets3/balanced_l0.nnue"
OPTS="$OPTS option.EvalFile2=$R/nets3/balanced_l1.nnue"
OPTS="$OPTS option.EvalFile3=$R/nets3/balanced_l2.nnue"
OPTS="$OPTS option.EvalFile4=$R/nets3/balanced_l3.nnue"
OPTS="$OPTS option.EvalFile5=$R/nets3/nvb.nnue"
OPTS="$OPTS option.EvalFile6=$R/nets3/nvr.nnue"
OPTS="$OPTS option.EvalFile7=$R/nets3/bvr.nnue"
OPTS="$OPTS option.EvalFile8=$R/nets3/rv2m.nnue"
OPTS="$OPTS option.EvalFile9=$R/nets3/qvmat.nnue"
OPTS="$OPTS option.EvalFile10=$R/nets3/oppb.nnue"
OPTS="$OPTS option.EvalFile11=$R/nets3/dvoretsky.nnue"
OPTS="$OPTS option.EvalFile12=$R/nets3/exchanges.nnue"
OPTS="$OPTS option.EvalFile13=$R/nets2/tb_training.nnue"

BATCH=${BATCH:-100}     # games per fastchess invocation
ST=${ST:-1}             # seconds per move
CONC=${CONC:-3}         # parallel matches

n=0
while true; do
  T=$(date -u +%Y%m%d_%H%M%S)
  for i in $(seq 1 $CONC); do
    ./fastchess-linux-x86-64/fastchess \
      -engine name=moe13a cmd=$A/fork_default.sh $OPTS \
      -engine name=moe13b cmd=$A/fork_default.sh $OPTS \
      -each proto=uci st=$ST -rounds $((BATCH/2)) -repeat \
      -openings file=$R/otb_book.epd format=epd order=random \
      -pgnout file=$OUT/games/sp_${T}_${i}.pgn \
      >> $OUT/games/sp_${T}_${i}.log 2>&1 &
    n=$((n+BATCH))
  done
  wait
  echo "$(date -u) batch done, total games: $n" >> $OUT/progress.log
done
