#!/bin/bash
R=/mnt/cold-raid6/chess-audit
N=$R/nets
SF=/srv/workspace/flychess/src/Stockfish-act/src/stockfish
{
echo uci; sleep 1
echo "setoption name EvalFile value $N/balanced_l0.nnue"
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
sleep 3
echo "position fen 4k3/8/8/8/8/8/8/QK6 w - - 0 1"
echo "go depth 1"; sleep 3
echo "position fen r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4"
echo "go depth 1"; sleep 3
echo "position fen r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 4 8"
echo "go depth 1"; sleep 3
echo quit; sleep 1
} | $SF 2>&1 | grep -E "score cp|bestmove" | head -6
