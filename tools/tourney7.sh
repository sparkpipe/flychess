#!/bin/bash
# 7-player round robin @ st=10: naked/stacked x Q/R2/R3 + e2850 anchor.
# Idempotent: pairings with a completed 6-game PGN are skipped on relaunch.
R=/mnt/cold-raid6/chess-audit
FC=/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess
FORK=/srv/workspace/flychess/src/Stockfish/src/stockfish
T=$R/tourney7
mkdir -p $T
HEADS="option.StackHead=$R/selfplay_rl/head_sQ.evh"

opts_for () {
  local N=$R/$1
  echo "option.EvalFile=$N/balanced_l0.nnue option.EvalFile2=$N/balanced_l1.nnue option.EvalFile3=$N/balanced_l2.nnue option.EvalFile4=$N/balanced_l3.nnue option.EvalFile5=$N/nvb.nnue option.EvalFile6=$N/nvr.nnue option.EvalFile7=$N/bvr.nnue option.EvalFile8=$N/rv2m.nnue option.EvalFile9=$N/qvmat.nnue option.EvalFile10=$N/oppb.nnue option.EvalFile11=$N/dvoretsky.nnue option.EvalFile12=$N/exchanges.nnue option.EvalFile13=$N/tactics.nnue"
}

spec_of () {
  case $1 in
    nQ)  echo "cmd=$FORK $(opts_for nets)";;
    nR2) echo "cmd=$FORK $(opts_for nets2)";;
    nR3) echo "cmd=$FORK $(opts_for nets3)";;
    sQ)  echo "cmd=$FORK $(opts_for nets) option.StackHead=$R/selfplay_rl/head_sQ.evh";;
    sR2) echo "cmd=$FORK $(opts_for nets2) option.StackHead=$R/selfplay_rl/head_sR2.evh";;
    sR3) echo "cmd=$FORK $(opts_for nets3) option.StackHead=$R/selfplay_rl/head_sR3.evh";;
    e2850) echo "cmd=/usr/games/stockfish option.UCI_LimitStrength=true option.UCI_Elo=2850";;
  esac
}

pairing () {
  local A=$1 B=$2 P=$T/${A}_vs_${B}
  if [ -f "$P.pgn" ] && [ "$(grep -c '\[Result' $P.pgn)" -ge 6 ]; then
    echo "skip ${A}_vs_${B} (done)" >> $T/progress.log; return
  fi
  $FC -engine name=$A $(spec_of $A) \
      -engine name=$B $(spec_of $B) \
      -each proto=uci st=10 timemargin=200 -rounds 3 -repeat \
      -openings file=$R/3open.epd format=epd \
      -pgnout file=$P.pgn > $P.log 2>&1
}

CANDS="nQ nR2 nR3 sQ sR2 sR3 e2850"
JOBS=0
for A in $CANDS; do
  for B in $CANDS; do
    if [ "$A" != "$B" ] && [[ ! -f $T/.done_${A}_${B} && ! -f $T/.done_${B}_${A} ]] && [[ "$A" < "$B" ]]; then
      pairing $A $B &
      JOBS=$((JOBS+1))
      if [ $((JOBS % 6)) -eq 0 ]; then wait; fi
    fi
  done
done
wait
echo "TOURNEY COMPLETE $(date)" >> $T/progress.log
