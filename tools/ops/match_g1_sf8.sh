#!/bin/bash
# general1 (2s/move) vs SF8 (1s/move) — standard 6-game match
: "${CHESS_DATA:?CHESS_DATA not set (data root)}"
: "${FASTCHESS_BIN:?FASTCHESS_BIN not set (fastchess binary)}"
: "${FORK_BIN:?FORK_BIN not set (n16-fork stockfish)}"
FC="$FASTCHESS_BIN"
GEN="$FORK_BIN"
GENN=$CHESS_DATA/runs-general1/best.nnue
SF8="$CHESS_DATA/engines/sf8/stockfish"  # rebuild+materialize before use
OUT=$CHESS_DATA/matches
$FC \
  -engine name=general1 cmd=$GEN st=2 option.EvalFile=$GENN \
  -engine name=SF8 cmd=$SF8 st=1 \
  -each proto=uci timemargin=200 \
  -rounds 3 -repeat -recover \
  -openings file="$CHESS_DATA/engines/3open.epd" format=epd \
  -pgnout file=$OUT/general1_2s_vs_sf8_1s.pgn \
  2>&1 | tee $OUT/general1_2s_vs_sf8_1s.log
echo MATCH-DONE
