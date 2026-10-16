#!/bin/bash
# general1 (single net, vanilla SF17) vs n16base (16-slot MoE) — 1s/move, 6 games
: "${CHESS_DATA:?CHESS_DATA not set (data root)}"
: "${FASTCHESS_BIN:?FASTCHESS_BIN not set (fastchess binary)}"
: "${FORK_BIN:?FORK_BIN not set (n16-fork stockfish)}"
FC="$FASTCHESS_BIN"
GEN="$FORK_BIN"
GENN=$CHESS_DATA/runs-general1/best.nnue
N16="$FORK_BIN"
E=$CHESS_DATA/engine16-baseline
OUT=$CHESS_DATA/matches
N16F="option.EvalFile=$E/tb.nnue option.EvalFile2=$E/mvr.nnue option.EvalFile3=$E/rv2m.nnue option.EvalFile4=$E/qvmat.nnue option.EvalFile5=$E/nvb.nnue option.EvalFile6=$E/piece_down.nnue option.EvalFile7=$E/oppb.nnue option.EvalFile8=$E/dv_Q.nnue option.EvalFile9=$E/dv_R.nnue option.EvalFile10=$E/dv_rest.nnue option.EvalFile11=$E/op_pawnimb.nnue option.EvalFile12=$E/op_even_l0.nnue option.EvalFile13=$E/op_even_l1.nnue option.EvalFile14=$E/op_even_l2p.nnue option.EvalFile15=$E/mg_unsafe.nnue option.EvalFile16=$E/mg_safe.nnue"
$FC \
  -engine name=general1 cmd=$GEN st=1 option.EvalFile=$GENN \
  -engine name=n16 cmd=$N16 st=1 $N16F \
  -each proto=uci timemargin=200 \
  -rounds 3 -repeat -recover \
  -openings file="$CHESS_DATA/engines/3open.epd" format=epd \
  -pgnout file=$OUT/general1_vs_n16_1v1.pgn \
  2>&1 | tee $OUT/general1_vs_n16_1v1.log
