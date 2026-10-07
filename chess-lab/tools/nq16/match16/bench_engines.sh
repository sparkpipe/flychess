#!/bin/bash
# NPS comparison: nQ13 vs n16, identical bench (16MB hash, 1 thread, depth 13).
NQ=/mnt/cold-raid6/chess-audit/nq13_binary_backup
NQN=/mnt/cold-raid6/chess-audit/nets
N16=/srv/workspace/flychess/src/Stockfish/src/stockfish
E16=/srv/workspace/chess-active/engine16

NQ_OPTS="setoption name EvalFile value $NQN/balanced_l0.nnue
setoption name EvalFile2 value $NQN/balanced_l1.nnue
setoption name EvalFile3 value $NQN/balanced_l2.nnue
setoption name EvalFile4 value $NQN/balanced_l3.nnue
setoption name EvalFile5 value $NQN/nvb.nnue
setoption name EvalFile6 value $NQN/nvr.nnue
setoption name EvalFile7 value $NQN/bvr.nnue
setoption name EvalFile8 value $NQN/rv2m.nnue
setoption name EvalFile9 value $NQN/qvmat.nnue
setoption name EvalFile10 value $NQN/oppb.nnue
setoption name EvalFile11 value $NQN/dvoretsky.nnue
setoption name EvalFile12 value $NQN/exchanges.nnue
setoption name EvalFile13 value $NQN/tactics.nnue"

N16_OPTS="setoption name EvalFile value $E16/tb.nnue
setoption name EvalFile2 value $E16/mvr.nnue
setoption name EvalFile3 value $E16/rv2m.nnue
setoption name EvalFile4 value $E16/qvmat.nnue
setoption name EvalFile5 value $E16/nvb.nnue
setoption name EvalFile6 value $E16/piece_down.nnue
setoption name EvalFile7 value $E16/oppb.nnue
setoption name EvalFile8 value $E16/dv_Q.nnue
setoption name EvalFile9 value $E16/dv_R.nnue
setoption name EvalFile10 value $E16/dv_rest.nnue
setoption name EvalFile11 value $E16/op_pawnimb.nnue
setoption name EvalFile12 value $E16/op_even_l0.nnue
setoption name EvalFile13 value $E16/op_even_l1.nnue
setoption name EvalFile14 value $E16/op_even_l2p.nnue
setoption name EvalFile15 value $E16/mg_unsafe.nnue
setoption name EvalFile16 value $E16/mg_safe.nnue"

echo "=== smoke: n16 routing + depth-10 search on startpos"
{ echo "$N16_OPTS"; echo "position startpos"; echo "go depth 10"; sleep 12; echo quit; } | $N16 2>/dev/null | grep -E "^(route|info depth 10 |bestmove)" | head -6

for run in 1 2; do
  echo "=== nQ13 bench run $run"
  { echo "$NQ_OPTS"; echo "bench 16 1 13 default depth"; echo quit; } | $NQ 2>/dev/null | grep -E "Nodes searched|Total time|nps" | head -4
  echo "=== n16 bench run $run"
  { echo "$N16_OPTS"; echo "bench 16 1 13 default depth"; echo quit; } | $N16 2>/dev/null | grep -E "Nodes searched|Total time|nps" | head -4
done
