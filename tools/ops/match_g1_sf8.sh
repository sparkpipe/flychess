#!/bin/bash
# general1 (2s/move) vs SF8 (1s/move) — standard 6-game match
FC=/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess
GEN=/srv/workspace/flychess/src/Stockfish/src/stockfish
GENN=/srv/workspace/chess-active/runs-general1/best.nnue
SF8=/tmp/sf8src/src/stockfish
OUT=/srv/workspace/chess-active/matches
$FC \
  -engine name=general1 cmd=$GEN st=2 option.EvalFile=$GENN \
  -engine name=SF8 cmd=$SF8 st=1 \
  -each proto=uci timemargin=200 \
  -rounds 3 -repeat -recover \
  -openings file=/mnt/cold-raid6/chess-audit/3open.epd format=epd \
  -pgnout file=$OUT/general1_2s_vs_sf8_1s.pgn \
  2>&1 | tee $OUT/general1_2s_vs_sf8_1s.log
echo MATCH-DONE
