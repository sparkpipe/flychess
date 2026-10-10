#!/bin/bash
MT=${1:-1}
cd ~/flychess
NETS=$HOME/flychess/nets
B=$HOME/flychess/Stockfish/src/stockfish
OPTS="option.EvalFile=$NETS/balanced_l0.nnue option.EvalFile2=$NETS/balanced_l1.nnue option.EvalFile3=$NETS/balanced_l2.nnue option.EvalFile4=$NETS/balanced_l3.nnue option.EvalFile5=$NETS/nvb.nnue option.EvalFile6=$NETS/nvr.nnue option.EvalFile7=$NETS/bvr.nnue option.EvalFile8=$NETS/rv2m.nnue option.EvalFile9=$NETS/qvmat.nnue option.EvalFile10=$NETS/oppb.nnue option.EvalFile11=$NETS/dvoretsky.nnue option.EvalFile12=$NETS/exchanges.nnue option.EvalFile13=$NETS/tactics.nnue option.StackHead=$HOME/flychess/head.evh"
exec ./fastchess -engine name=sp_a cmd=$B $OPTS \
  -engine name=sp_b cmd=$B $OPTS \
  -each proto=uci st=$MT timemargin=200 -rounds 64 -repeat \
  -openings file=shard.epd format=epd order=sequential \
  -pgnout file=games_iter2.pgn -concurrency 16
