#!/bin/bash
# general3B matches: vs general3, general2, SF8, SF17 at 2s/move and depth-20.
: "${CHESS_DATA:?CHESS_DATA not set (data root)}"
: "${FASTCHESS_BIN:?FASTCHESS_BIN not set (fastchess binary)}"
: "${FORK_BIN:?FORK_BIN not set (n16-fork stockfish)}"
: "${SF17_BIN:?SF17_BIN not set (sf17 binary)}"
cd $CHESS_DATA
# wait for general3B to finish training
while ! ls runs-general3b/best.nnue > /dev/null 2>&1; do
  sleep 120
done
echo "[$(date +%H:%M)] general3B best ready — starting matches"

FC="$FASTCHESS_BIN"
N16="$FORK_BIN"
SF8="$CHESS_DATA/engines/sf8/stockfish"  # rebuild+materialize before use
SF17="$SF17_BIN"
OUT=$CHESS_DATA/matches

G3B="option.EvalFile=$CHESS_DATA/runs-general3b/best.nnue"
G3="option.EvalFile=$CHESS_DATA/runs-general3/best.nnue"
G2="option.EvalFile=$CHESS_DATA/runs-general2/best.nnue"

# 2s/move matches
for OPP in general3 general2 SF8 SF17; do
  case $OPP in
    general3) OCMD=$N16;   OOPT=$G3 ;;
    general2) OCMD=$N16;   OOPT=$G2 ;;
    SF8)      OCMD=$SF8;   OOPT="" ;;
    SF17)     OCMD=$SF17;  OOPT="" ;;
  esac
  rm -f $OUT/general3b_vs_${OPP}_2s.pgn
  $FC \
    -engine name=general3B cmd=$N16 st=2 $G3B \
    -engine name=$OPP cmd=$OCMD st=2 $OOPT \
    -each proto=uci timemargin=200 \
    -rounds 3 -repeat -recover \
    -openings file="$CHESS_DATA/engines/3open.epd" format=epd \
    -pgnout file=$OUT/general3b_vs_${OPP}_2s.pgn \
    2>&1 | tee $OUT/general3b_vs_${OPP}_2s.log
  echo "=== g3B vs $OPP 2s: $(grep Points $OUT/general3b_vs_${OPP}_2s.log | tail -1) ==="
done

# depth-20 matches
for OPP in general3 general2 SF8 SF17; do
  case $OPP in
    general3) OCMD=$N16;   OOPT=$G3 ;;
    general2) OCMD=$N16;   OOPT=$G2 ;;
    SF8)      OCMD=$SF8;   OOPT="" ;;
    SF17)     OCMD=$SF17;  OOPT="" ;;
  esac
  rm -f $OUT/general3b_vs_${OPP}_d20.pgn
  $FC \
    -engine name=general3B cmd=$N16 depth=20 $G3B \
    -engine name=$OPP cmd=$OCMD depth=20 $OOPT \
    -each proto=uci \
    -rounds 3 -repeat -recover \
    -openings file="$CHESS_DATA/engines/3open.epd" format=epd \
    -pgnout file=$OUT/general3b_vs_${OPP}_d20.pgn \
    2>&1 | tee $OUT/general3b_vs_${OPP}_d20.log
  echo "=== g3B vs $OPP d20: $(grep Points $OUT/general3b_vs_${OPP}_d20.log | tail -1) ==="
done
echo ALL-GENERAL3B-MATCHES-DONE
