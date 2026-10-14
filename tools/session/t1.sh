#!/bin/bash
# Box fastchess sanity test, output on real disk (not tmpfs).
FC=/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess
SF8=/tmp/sf8src/src/stockfish
mkdir -p /srv/workspace/chess-active/matchtest
$FC -engine name=a cmd=$SF8 st=1 -engine name=b cmd=$SF8 st=1 \
  -each proto=uci timemargin=200 -rounds 1 \
  > /srv/workspace/chess-active/matchtest/t1.log 2>&1
echo DONE >> /srv/workspace/chess-active/matchtest/t1.log
