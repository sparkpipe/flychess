#!/bin/bash
# general1 (single net) vs nQ13 — 1s/move then 10s/move, standard 6-game matches
FC=/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess
GEN=/srv/workspace/flychess/src/Stockfish/src/stockfish
GENN=/srv/workspace/chess-active/runs-general1/best.nnue
NQ=/mnt/cold-raid6/chess-audit/nq13_binary_backup
NQN=/mnt/cold-raid6/chess-audit/nets
OUT=/srv/workspace/chess-active/matches
NQF="option.EvalFile=$NQN/balanced_l0.nnue option.EvalFile2=$NQN/balanced_l1.nnue option.EvalFile3=$NQN/balanced_l2.nnue option.EvalFile4=$NQN/balanced_l3.nnue option.EvalFile5=$NQN/nvb.nnue option.EvalFile6=$NQN/nvr.nnue option.EvalFile7=$NQN/bvr.nnue option.EvalFile8=$NQN/rv2m.nnue option.EvalFile9=$NQN/qvmat.nnue option.EvalFile10=$NQN/oppb.nnue option.EvalFile11=$NQN/dvoretsky.nnue option.EvalFile12=$NQN/exchanges.nnue option.EvalFile13=$NQN/tactics.nnue"
for ST in 1 10; do
  $FC \
    -engine name=general1 cmd=$GEN st=$ST option.EvalFile=$GENN \
    -engine name=nQ cmd=$NQ st=$ST $NQF \
    -each proto=uci timemargin=200 \
    -rounds 3 -repeat -recover \
    -openings file=/mnt/cold-raid6/chess-audit/3open.epd format=epd \
    -pgnout file=$OUT/general1_vs_nq_${ST}s.pgn \
    2>&1 | tee $OUT/general1_vs_nq_${ST}s.log
done
echo BOTH-MATCHES-DONE
