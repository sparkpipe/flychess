#!/bin/bash
R=/mnt/cold-raid6/chess-audit
N=$R/nets/balanced_l0.nnue
{
echo uci; sleep 1
for i in "" 2 3 4 5 6 7 8 9 10 11 12 13; do echo "setoption name EvalFile$i value $N"; done
sleep 2
echo "position fen 4k3/8/8/8/8/8/8/QK6 b - - 0 1"
echo eval; sleep 2
echo "position fen 4k3/8/8/8/8/8/8/QK6 w - - 0 1"
echo eval; sleep 2
echo quit; sleep 1
} | /srv/workspace/flychess/src/Stockfish/src/stockfish 2>&1
