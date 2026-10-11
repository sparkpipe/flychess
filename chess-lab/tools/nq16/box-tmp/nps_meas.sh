#!/bin/bash
# NPS: fixed 9-position mix at depth 11, both engines, identical settings.
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

FENS=(
"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
"rnbqkbnr/ppp1pppp/8/3p4/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"
"rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3"
"8/8/8/8/8/4k3/8/4K2R w K - 0 1"
"8/8/8/4k3/8/8/4n3/R3K3 w - - 0 1"
"8/5k2/8/4p3/8/8/4K3/5R2 w - - 0 1"
"4k3/2b5/8/4p3/8/3P4/2B5/4K3 w - - 0 1"
"4k3/8/8/8/8/2n2r2/8/2Q1K3 w - - 0 1"
"8/8/8/4k3/8/8/4n3/R3K2R w K - 0 1"
)

measure() { # $1=label $2=binary $3=opts
  local cmd=""
  for f in "${FENS[@]}"; do cmd+="position fen $f\ngo depth 11\n"; done
  { echo "$3"; printf "$cmd"; echo quit; } | $2 2>/dev/null \
    | grep -E "^info depth 11 .* nodes .* time " | awk -v L="$1" '
      { for (i=1;i<NF;i++) { if ($i=="nodes") n=$(i+1); if ($i=="time") t=$(i+1) }
        tn+=n; tt+=t; c++ }
      END { if (tt>0) printf "%s: %d positions, %d nodes, %d ms, %.0f nps\n", L, c, tn, tt, tn*1000/tt; else print L": NO DATA" }'
}

measure nQ13 "$NQ" "$NQ_OPTS"
measure n16 "$N16" "$N16_OPTS"
