#!/bin/bash
# Eval sign test: positions with known truth through both engines (match wiring).
# SF prints stm-pov scores. Expected column = stm-pov expectation.
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

declare -a NAMES=(T1_startpos T2_wUpQ_w T3_wUpQ_b T4_wDownR_w T5_wDownR_b T6_KRK_w T7_bUpQ_b)
declare -a FENS=(
"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
"rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
"rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w kq - 0 1"
"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR b kq - 0 1"
"8/8/8/8/8/4k3/8/4K2R w K - 0 1"
"rnbqkbnr/pppp1ppp/8/4p3/8/5PP1/PPPPP2P/RNBQKBNR b KQkq - 0 2"
)
declare -a EXPECT=(~0 BIG+ BIG- BIG- BIG+ BIG+ BIG-)

run() { # $1 label $2 binary $3 opts
  echo "== $1"
  for i in "${!FENS[@]}"; do
    s=$({ echo "$3"; echo "position fen ${FENS[$i]}"; echo "go depth 6"; sleep 1.2; echo quit; } | $2 2>/dev/null \
      | grep -E "^info depth 6 " | tail -1 | grep -oE "score cp [-0-9]+|score mate [-0-9]+" | head -1)
    echo "  ${NAMES[$i]}: expect ${EXPECT[$i]} -> $s"
  done
}
run nQ13 "$NQ" "$NQ_OPTS"
run n16  "$N16" "$N16_OPTS"
