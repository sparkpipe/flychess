#!/bin/bash
# n16 (quick nets) 2s/move vs nQ13 1s/move — standard 6-game match.
FC=/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess
NQ=/mnt/cold-raid6/chess-audit/nq13_binary_backup
NQN=/mnt/cold-raid6/chess-audit/nets
N16=/srv/workspace/flychess/src/Stockfish/src/stockfish
E=/srv/workspace/chess-active/engine16quick
OUT=/srv/workspace/chess-active/matches
mkdir -p $OUT

NQF="option.EvalFile=$NQN/balanced_l0.nnue option.EvalFile2=$NQN/balanced_l1.nnue option.EvalFile3=$NQN/balanced_l2.nnue option.EvalFile4=$NQN/balanced_l3.nnue option.EvalFile5=$NQN/nvb.nnue option.EvalFile6=$NQN/nvr.nnue option.EvalFile7=$NQN/bvr.nnue option.EvalFile8=$NQN/rv2m.nnue option.EvalFile9=$NQN/qvmat.nnue option.EvalFile10=$NQN/oppb.nnue option.EvalFile11=$NQN/dvoretsky.nnue option.EvalFile12=$NQN/exchanges.nnue option.EvalFile13=$NQN/tactics.nnue"

N16F="option.EvalFile=$E/tb.nnue option.EvalFile2=$E/mvr.nnue option.EvalFile3=$E/rv2m.nnue option.EvalFile4=$E/qvmat.nnue option.EvalFile5=$E/nvb.nnue option.EvalFile6=$E/piece_down.nnue option.EvalFile7=$E/oppb.nnue option.EvalFile8=$E/dv_Q.nnue option.EvalFile9=$E/dv_R.nnue option.EvalFile10=$E/dv_rest.nnue option.EvalFile11=$E/op_pawnimb.nnue option.EvalFile12=$E/op_even_l0.nnue option.EvalFile13=$E/op_even_l1.nnue option.EvalFile14=$E/op_even_l2p.nnue option.EvalFile15=$E/mg_unsafe.nnue option.EvalFile16=$E/mg_safe.nnue"

$FC \
  -engine name=n16quick cmd=$N16 st=2 $N16F \
  -engine name=nQ cmd=$NQ st=1 $NQF \
  -each proto=uci timemargin=200 \
  -rounds 3 -repeat -recover \
  -openings file=/mnt/cold-raid6/chess-audit/3open.epd format=epd \
  -pgnout file=$OUT/n16quick_vs_nq_2v1.pgn \
  2>&1 | tee $OUT/n16quick_vs_nq_2v1.log
