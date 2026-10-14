#!/bin/bash
# general2 vs nQ at FIXED DEPTH 20 — no time management, pure search quality.
cd /srv/workspace/chess-active
FC=/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess
N16=/srv/workspace/flychess/src/Stockfish/src/stockfish
NQ=/mnt/cold-raid6/chess-audit/nq13_binary_backup
NQN=/mnt/cold-raid6/chess-audit/nets
GEN2=/srv/workspace/chess-active/runs-general2/best.nnue
OUT=/srv/workspace/chess-active/matches

G2F="option.EvalFile=$GEN2"
NQF="option.EvalFile=$NQN/balanced_l0.nnue option.EvalFile2=$NQN/balanced_l1.nnue option.EvalFile3=$NQN/balanced_l2.nnue option.EvalFile4=$NQN/balanced_l3.nnue option.EvalFile5=$NQN/nvb.nnue option.EvalFile6=$NQN/nvr.nnue option.EvalFile7=$NQN/bvr.nnue option.EvalFile8=$NQN/rv2m.nnue option.EvalFile9=$NQN/qvmat.nnue option.EvalFile10=$NQN/oppb.nnue option.EvalFile11=$NQN/dvoretsky.nnue option.EvalFile12=$NQN/exchanges.nnue option.EvalFile13=$NQN/tactics.nnue"

rm -f $OUT/general2_vs_nQ_d20.pgn
$FC \
  -engine name=general2 cmd=$N16 depth=20 $G2F \
  -engine name=nQ cmd=$NQ depth=20 $NQF \
  -each proto=uci \
  -rounds 3 -repeat -recover \
  -openings file=/mnt/cold-raid6/chess-audit/3open.epd format=epd \
  -pgnout file=$OUT/general2_vs_nQ_d20.pgn \
  2>&1 | tee $OUT/general2_vs_nQ_d20.log
echo "D20 MATCH DONE"
