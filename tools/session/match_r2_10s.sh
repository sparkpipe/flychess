#!/bin/bash
# correct16r2 10s/move matches vs nQ16, nQ, general1, SF8
cd /srv/workspace/chess-active
FC=/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess
N16=/srv/workspace/flychess/src/Stockfish/src/stockfish
NQ=/mnt/cold-raid6/chess-audit/nq13_binary_backup
NQN=/mnt/cold-raid6/chess-audit/nets
GENN=/srv/workspace/chess-active/runs-general1/best.nnue
SF8=/tmp/sf8src/src/stockfish
EB=/srv/workspace/chess-active/engine16-baseline
E=/srv/workspace/chess-active/engine16-correct-r2
OUT=/srv/workspace/chess-active/matches

C16F="option.EvalFile=$E/tb.nnue option.EvalFile2=$E/mvr.nnue option.EvalFile3=$E/rv2m.nnue option.EvalFile4=$E/qvmat.nnue option.EvalFile5=$E/nvb.nnue option.EvalFile6=$E/piece_down.nnue option.EvalFile7=$E/oppb.nnue option.EvalFile8=$E/dv_Q.nnue option.EvalFile9=$E/dv_R.nnue option.EvalFile10=$E/dv_rest.nnue option.EvalFile11=$E/op_pawnimb.nnue option.EvalFile12=$E/op_even_l0.nnue option.EvalFile13=$E/op_even_l1.nnue option.EvalFile14=$E/op_even_l2p.nnue option.EvalFile15=$E/mg_unsafe.nnue option.EvalFile16=$E/mg_safe.nnue"

NQ16F="option.EvalFile=$EB/tb.nnue option.EvalFile2=$EB/mvr.nnue option.EvalFile3=$EB/rv2m.nnue option.EvalFile4=$EB/qvmat.nnue option.EvalFile5=$EB/nvb.nnue option.EvalFile6=$EB/piece_down.nnue option.EvalFile7=$EB/oppb.nnue option.EvalFile8=$EB/dv_Q.nnue option.EvalFile9=$EB/dv_R.nnue option.EvalFile10=$EB/dv_rest.nnue option.EvalFile11=$EB/op_pawnimb.nnue option.EvalFile12=$EB/op_even_l0.nnue option.EvalFile13=$EB/op_even_l1.nnue option.EvalFile14=$EB/op_even_l2p.nnue option.EvalFile15=$EB/mg_unsafe.nnue option.EvalFile16=$EB/mg_safe.nnue"

NQF="option.EvalFile=$NQN/balanced_l0.nnue option.EvalFile2=$NQN/balanced_l1.nnue option.EvalFile3=$NQN/balanced_l2.nnue option.EvalFile4=$NQN/balanced_l3.nnue option.EvalFile5=$NQN/nvb.nnue option.EvalFile6=$NQN/nvr.nnue option.EvalFile7=$NQN/bvr.nnue option.EvalFile8=$NQN/rv2m.nnue option.EvalFile9=$NQN/qvmat.nnue option.EvalFile10=$NQN/oppb.nnue option.EvalFile11=$NQN/dvoretsky.nnue option.EvalFile12=$NQN/exchanges.nnue option.EvalFile13=$NQN/tactics.nnue"

for OPP in nQ16 nQ general1 SF8; do
  case $OPP in
    nQ16)     OCMD=$N16; OOPT=$NQ16F ;;
    nQ)       OCMD=$NQ;  OOPT=$NQF ;;
    general1) OCMD=$N16; OOPT="option.EvalFile=$GENN" ;;
    SF8)      OCMD=$SF8; OOPT="" ;;
  esac
  rm -f $OUT/c16r2_vs_${OPP}_10s.pgn
  $FC \
    -engine name=correct16r2 cmd=$N16 st=10 $C16F \
    -engine name=$OPP cmd=$OCMD st=10 $OOPT \
    -each proto=uci timemargin=200 \
    -rounds 3 -repeat -recover \
    -openings file=/mnt/cold-raid6/chess-audit/3open.epd format=epd \
    -pgnout file=$OUT/c16r2_vs_${OPP}_10s.pgn \
    2>&1 | tee $OUT/c16r2_vs_${OPP}_10s.log
  echo "MATCH $OPP 10s DONE"
done
echo ALL-10S-MATCHES-DONE
