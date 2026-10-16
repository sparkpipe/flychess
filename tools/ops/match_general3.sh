#!/bin/bash
# general3 matches vs all references at 1s, then 10s.
cd /srv/workspace/chess-active
FC=/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess
N16=/srv/workspace/flychess/src/Stockfish/src/stockfish
NQ=/mnt/cold-raid6/chess-audit/nq13_binary_backup
NQN=/mnt/cold-raid6/chess-audit/nets
G3=/srv/workspace/chess-active/runs-general3/best.nnue
SF8=/tmp/sf8src/src/stockfish
SF17=/srv/workspace/flychess/src/sf17/src/stockfish
ER2=/srv/workspace/chess-active/engine16-correct-r2
GEN2=/srv/workspace/chess-active/runs-general2/best.nnue
OUT=/srv/workspace/chess-active/matches

G3F="option.EvalFile=$G3"
GEN2F="option.EvalFile=$GEN2"
C16R2F="option.EvalFile=$ER2/tb.nnue option.EvalFile2=$ER2/mvr.nnue option.EvalFile3=$ER2/rv2m.nnue option.EvalFile4=$ER2/qvmat.nnue option.EvalFile5=$ER2/nvb.nnue option.EvalFile6=$ER2/piece_down.nnue option.EvalFile7=$ER2/oppb.nnue option.EvalFile8=$ER2/dv_Q.nnue option.EvalFile9=$ER2/dv_R.nnue option.EvalFile10=$ER2/dv_rest.nnue option.EvalFile11=$ER2/op_pawnimb.nnue option.EvalFile12=$ER2/op_even_l0.nnue option.EvalFile13=$ER2/op_even_l1.nnue option.EvalFile14=$ER2/op_even_l2p.nnue option.EvalFile15=$ER2/mg_unsafe.nnue option.EvalFile16=$ER2/mg_safe.nnue"
NQF="option.EvalFile=$NQN/balanced_l0.nnue option.EvalFile2=$NQN/balanced_l1.nnue option.EvalFile3=$NQN/balanced_l2.nnue option.EvalFile4=$NQN/balanced_l3.nnue option.EvalFile5=$NQN/nvb.nnue option.EvalFile6=$NQN/nvr.nnue option.EvalFile7=$NQN/bvr.nnue option.EvalFile8=$NQN/rv2m.nnue option.EvalFile9=$NQN/qvmat.nnue option.EvalFile10=$NQN/oppb.nnue option.EvalFile11=$NQN/dvoretsky.nnue option.EvalFile12=$NQN/exchanges.nnue option.EvalFile13=$NQN/tactics.nnue"

for ST in 1 10; do
  for OPP in general2 correct16r2 nQ SF8 SF17; do
    case $OPP in
      general2)    OCMD=$N16;  OOPT=$GEN2F ;;
      correct16r2) OCMD=$N16;  OOPT=$C16R2F ;;
      nQ)          OCMD=$NQ;   OOPT=$NQF ;;
      SF8)         OCMD=$SF8;  OOPT="" ;;
      SF17)        OCMD=$SF17; OOPT="" ;;
    esac
    rm -f $OUT/general3_vs_${OPP}_${ST}s.pgn
    $FC \
      -engine name=general3 cmd=$N16 st=$ST $G3F \
      -engine name=$OPP cmd=$OCMD st=$ST $OOPT \
      -each proto=uci timemargin=200 \
      -rounds 3 -repeat -recover \
      -openings file=/mnt/cold-raid6/chess-audit/3open.epd format=epd \
      -pgnout file=$OUT/general3_vs_${OPP}_${ST}s.pgn \
      2>&1 | tee $OUT/general3_vs_${OPP}_${ST}s.log
    echo "=== general3 vs $OPP ${ST}s: $(grep Points $OUT/general3_vs_${OPP}_${ST}s.log | tail -1) ==="
  done
done
echo ALL-GENERAL3-MATCHES-DONE
