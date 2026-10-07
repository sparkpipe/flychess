#!/bin/bash
R=/mnt/cold-raid6/chess-audit
N=$R/nets
SF=/srv/workspace/flychess/src/Stockfish-act/src/stockfish
{
echo uci; sleep 1
for i in "" 2 3 4 5 6 7 8 9 10 11 12 13; do
  echo "setoption name EvalFile$i value $N/balanced_l0.nnue"
done
# set all slots properly
echo "setoption name EvalFile2 value $N/balanced_l1.nnue"
echo "setoption name EvalFile3 value $N/balanced_l2.nnue"
echo "setoption name EvalFile4 value $N/balanced_l3.nnue"
echo "setoption name EvalFile5 value $N/nvb.nnue"
echo "setoption name EvalFile6 value $N/nvr.nnue"
echo "setoption name EvalFile7 value $N/bvr.nnue"
echo "setoption name EvalFile8 value $N/rv2m.nnue"
echo "setoption name EvalFile9 value $N/qvmat.nnue"
echo "setoption name EvalFile10 value $N/oppb.nnue"
echo "setoption name EvalFile11 value $N/dvoretsky.nnue"
echo "setoption name EvalFile12 value $N/exchanges.nnue"
echo "setoption name EvalFile13 value $N/tactics.nnue"
echo "setoption name StackHead value $R/selfplay_rl/head_dense_v3.evh"
sleep 2
echo "position fen 4k3/8/8/8/8/8/8/QK6 w - - 0 1"
echo "go depth 1"; sleep 2
echo "position fen 4k3/8/8/8/8/8/8/qk6 b - - 0 1"
echo "go depth 1"; sleep 2
echo "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
echo "go depth 1"; sleep 2
echo quit; sleep 1
} | $SF 2>&1 | grep -E "StackHead|score cp" | head -6
