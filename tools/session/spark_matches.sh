#!/bin/bash
# The four correct16 matches, on spark0 (ARM, clean node).
# Note: SF8 has no ARM build here; general1/nQ/nQ16 all run on the ARM
# 16-slot fork. SF8 match is skipped with a note (would need ARM SF8 build).
cd /home/spark0/c16match
FC=/home/spark0/flychess/fastchess
N16=/home/spark0/n16src/src/stockfish
OUT=/home/spark0/c16match/results
mkdir -p $OUT

C16F="option.EvalFile=$PWD/correct/tb.nnue option.EvalFile2=$PWD/correct/mvr.nnue option.EvalFile3=$PWD/correct/rv2m.nnue option.EvalFile4=$PWD/correct/qvmat.nnue option.EvalFile5=$PWD/correct/nvb.nnue option.EvalFile6=$PWD/correct/piece_down.nnue option.EvalFile7=$PWD/correct/oppb.nnue option.EvalFile8=$PWD/correct/dv_Q.nnue option.EvalFile9=$PWD/correct/dv_R.nnue option.EvalFile10=$PWD/correct/dv_rest.nnue option.EvalFile11=$PWD/correct/op_pawnimb.nnue option.EvalFile12=$PWD/correct/op_even_l0.nnue option.EvalFile13=$PWD/correct/op_even_l1.nnue option.EvalFile14=$PWD/correct/op_even_l2p.nnue option.EvalFile15=$PWD/correct/mg_unsafe.nnue option.EvalFile16=$PWD/correct/mg_safe.nnue"

NQ16F="option.EvalFile=$PWD/old/tb.nnue option.EvalFile2=$PWD/old/mvr.nnue option.EvalFile3=$PWD/old/rv2m.nnue option.EvalFile4=$PWD/old/qvmat.nnue option.EvalFile5=$PWD/old/nvb.nnue option.EvalFile6=$PWD/old/piece_down.nnue option.EvalFile7=$PWD/old/oppb.nnue option.EvalFile8=$PWD/old/dv_Q.nnue option.EvalFile9=$PWD/old/dv_R.nnue option.EvalFile10=$PWD/old/dv_rest.nnue option.EvalFile11=$PWD/old/op_pawnimb.nnue option.EvalFile12=$PWD/old/op_even_l0.nnue option.EvalFile13=$PWD/old/op_even_l1.nnue option.EvalFile14=$PWD/old/op_even_l2p.nnue option.EvalFile15=$PWD/old/mg_unsafe.nnue option.EvalFile16=$PWD/old/mg_safe.nnue"

NQF="option.EvalFile=$PWD/nq/balanced_l0.nnue option.EvalFile2=$PWD/nq/balanced_l1.nnue option.EvalFile3=$PWD/nq/balanced_l2.nnue option.EvalFile4=$PWD/nq/balanced_l3.nnue option.EvalFile5=$PWD/nq/nvb.nnue option.EvalFile6=$PWD/nq/nvr.nnue option.EvalFile7=$PWD/nq/bvr.nnue option.EvalFile8=$PWD/nq/rv2m.nnue option.EvalFile9=$PWD/nq/qvmat.nnue option.EvalFile10=$PWD/nq/oppb.nnue option.EvalFile11=$PWD/nq/dvoretsky.nnue option.EvalFile12=$PWD/nq/exchanges.nnue option.EvalFile13=$PWD/nq/tactics.nnue"

G1F="option.EvalFile=$PWD/general1.nnue"

for OPP in nQ16 nQ general1; do
  case $OPP in
    nQ16)     OOPT=$NQ16F ;;
    nQ)       OOPT=$NQF ;;
    general1) OOPT=$G1F ;;
  esac
  $FC \
    -engine name=correct16 cmd=$N16 st=1 $C16F \
    -engine name=$OPP cmd=$N16 st=1 $OOPT \
    -each proto=uci timemargin=200 \
    -rounds 3 -repeat -recover \
    -openings file=$PWD/3open.epd format=epd \
    -pgnout file=$OUT/correct16_vs_${OPP}_1s.pgn \
    2>&1 | tee $OUT/correct16_vs_${OPP}_1s.log
  echo "MATCH $OPP DONE"
done
echo ALL-SPARK-MATCHES-DONE
