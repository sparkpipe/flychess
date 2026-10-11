#!/bin/bash
# Standard 6-game match: nQ (1s/move) vs n16 (1.14s/move = nps-calibrated).
# 3 openings x color-reversed, timemargin 200, recover, PGN out.
FC=/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess
NQ=/mnt/cold-raid6/chess-audit/nq13_binary_backup
NQN=/mnt/cold-raid6/chess-audit/nets
N16=/srv/workspace/flychess/src/Stockfish/src/stockfish
E16=/srv/workspace/chess-active/engine16
OUT=/srv/workspace/chess-active/matches
EPD=/mnt/cold-raid6/chess-audit/3open.epd
mkdir -p $OUT

NQF="option.EvalFile=$NQN/balanced_l0.nnue option.EvalFile2=$NQN/balanced_l1.nnue option.EvalFile3=$NQN/balanced_l2.nnue option.EvalFile4=$NQN/balanced_l3.nnue option.EvalFile5=$NQN/nvb.nnue option.EvalFile6=$NQN/nvr.nnue option.EvalFile7=$NQN/bvr.nnue option.EvalFile8=$NQN/rv2m.nnue option.EvalFile9=$NQN/qvmat.nnue option.EvalFile10=$NQN/oppb.nnue option.EvalFile11=$NQN/dvoretsky.nnue option.EvalFile12=$NQN/exchanges.nnue option.EvalFile13=$NQN/tactics.nnue"

N16F="option.EvalFile=$E16/tb.nnue option.EvalFile2=$E16/mvr.nnue option.EvalFile3=$E16/rv2m.nnue option.EvalFile4=$E16/qvmat.nnue option.EvalFile5=$E16/nvb.nnue option.EvalFile6=$E16/piece_down.nnue option.EvalFile7=$E16/oppb.nnue option.EvalFile8=$E16/dv_Q.nnue option.EvalFile9=$E16/dv_R.nnue option.EvalFile10=$E16/dv_rest.nnue option.EvalFile11=$E16/op_pawnimb.nnue option.EvalFile12=$E16/op_even_l0.nnue option.EvalFile13=$E16/op_even_l1.nnue option.EvalFile14=$E16/op_even_l2p.nnue option.EvalFile15=$E16/mg_unsafe.nnue option.EvalFile16=$E16/mg_safe.nnue"

$FC \
  -engine name=nQ cmd=$NQ st=1 $NQF \
  -engine name=n16 cmd=$N16 st=1.14 $N16F \
  -each proto=uci timemargin=200 \
  -rounds 3 -repeat -recover \
  -openings file=$EPD format=epd \
  -pgnout file=$OUT/nq_vs_n16_1s.pgn \
  2>&1 | tee $OUT/nq_vs_n16_1s.log
