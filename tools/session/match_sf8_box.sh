#!/bin/bash
# correct16 vs SF8, 1s vs 1s, on the box (fastchess verified working after
# tmpfs clear). Output to real disk.
FC=/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess
N16=/srv/workspace/flychess/src/Stockfish/src/stockfish
SF8=/tmp/sf8src/src/stockfish
E=/srv/workspace/chess-active/engine16-correct
OUT=/srv/workspace/chess-active/matches

C16F="option.EvalFile=$E/tb.nnue option.EvalFile2=$E/mvr.nnue option.EvalFile3=$E/rv2m.nnue option.EvalFile4=$E/qvmat.nnue option.EvalFile5=$E/nvb.nnue option.EvalFile6=$E/piece_down.nnue option.EvalFile7=$E/oppb.nnue option.EvalFile8=$E/dv_Q.nnue option.EvalFile9=$E/dv_R.nnue option.EvalFile10=$E/dv_rest.nnue option.EvalFile11=$E/op_pawnimb.nnue option.EvalFile12=$E/op_even_l0.nnue option.EvalFile13=$E/op_even_l1.nnue option.EvalFile14=$E/op_even_l2p.nnue option.EvalFile15=$E/mg_unsafe.nnue option.EvalFile16=$E/mg_safe.nnue"

$FC \
  -engine name=correct16 cmd=$N16 st=1 $C16F \
  -engine name=SF8 cmd=$SF8 st=1 \
  -each proto=uci timemargin=200 \
  -rounds 3 -repeat -recover \
  -openings file=/mnt/cold-raid6/chess-audit/3open.epd format=epd \
  -pgnout file=$OUT/correct16_vs_SF8_1s.pgn \
  2>&1 | tee $OUT/correct16_vs_SF8_1s.log
echo "MATCH SF8 DONE"
