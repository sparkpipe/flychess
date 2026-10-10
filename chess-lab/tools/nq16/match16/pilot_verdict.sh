#!/bin/bash
# Pilot verdict: eval-sign test with pilot piece_down net in slot 6, rest = engine16.
N16=/srv/workspace/flychess/src/Stockfish/src/stockfish
E16=/srv/workspace/chess-active/engine16
P=$1  # pilot net path
N16_OPTS="setoption name EvalFile value $E16/tb.nnue
setoption name EvalFile2 value $E16/mvr.nnue
setoption name EvalFile3 value $E16/rv2m.nnue
setoption name EvalFile4 value $E16/qvmat.nnue
setoption name EvalFile5 value $E16/nvb.nnue
setoption name EvalFile6 value $P
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
declare -A F=(
[T2_wUpQ_w]="rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
[T3_wUpQ_b]="rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
[T4_wDownR_w]="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w kq - 0 1"
[T5_wDownR_b]="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR b kq - 0 1"
)
echo "== pilot net: $P"
for t in T2_wUpQ_w T3_wUpQ_b T4_wDownR_w T5_wDownR_b; do
  s=$({ echo "$N16_OPTS"; echo "position fen ${F[$t]}"; echo "go depth 6"; sleep 1.5; echo quit; } | $N16 2>/dev/null | grep "^info depth 6 " | tail -1 | grep -oE "score cp [-0-9]+|score mate [-0-9]+")
  echo "  $t -> $s"
done
