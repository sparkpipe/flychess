#!/bin/bash
# 6-candidate round robin: naked/stacked x quarter/round2/round3, st=10
R=/mnt/cold-raid6/chess-audit
FC=/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess
FORK=/srv/workspace/flychess/src/Stockfish/src/stockfish
T=$R/tourney6
mkdir -p $T
HEAD="option.StackHead=$R/selfplay_rl/head_fullstack_fixed.evh"

opts_for () {  # netsdir -> option string
  local N=$R/$1
  echo "option.EvalFile=$N/balanced_l0.nnue option.EvalFile2=$N/balanced_l1.nnue option.EvalFile3=$N/balanced_l2.nnue option.EvalFile4=$N/balanced_l3.nnue option.EvalFile5=$N/nvb.nnue option.EvalFile6=$N/nvr.nnue option.EvalFile7=$N/bvr.nnue option.EvalFile8=$N/rv2m.nnue option.EvalFile9=$N/qvmat.nnue option.EvalFile10=$N/oppb.nnue option.EvalFile11=$N/dvoretsky.nnue option.EvalFile12=$N/exchanges.nnue option.EvalFile13=$N/tactics.nnue"
}

declare -A COPTS
COPTS[nQ]="$(opts_for nets)"
COPTS[nR2]="$(opts_for nets2)"
COPTS[nR3]="$(opts_for nets3)"
COPTS[sQ]="${COPTS[nQ]} $HEAD"
COPTS[sR2]="${COPTS[nR2]} $HEAD"
COPTS[sR3]="${COPTS[nR3]} $HEAD"
CANDS="nQ nR2 nR3 sQ sR2 sR3"

pairing () {
  local A=$1 B=$2
  $FC -engine name=$A cmd=$FORK ${COPTS[$A]} \
      -engine name=$B cmd=$FORK ${COPTS[$B]} \
      -each proto=uci st=10 timemargin=200 -rounds 3 -repeat \
      -openings file=$R/3open.epd format=epd \
      -pgnout file=$T/${A}_vs_${B}.pgn > $T/${A}_vs_${B}.log 2>&1
}

JOBS=0
for A in $CANDS; do
  for B in $CANDS; do
    # A<B ordering: nQ<nR2<nR3<sQ<sR2<sR3 lexicographic works for these names
    if [[ "$A" < "$B" ]]; then
      pairing $A $B &
      JOBS=$((JOBS+1))
      if [ $((JOBS % 6)) -eq 0 ]; then wait; echo "batch of 6 done at $(date +%H:%M)" >> $T/progress.log; fi
    fi
  done
done
wait
echo "ALL 15 PAIRINGS DONE $(date)" >> $T/progress.log
