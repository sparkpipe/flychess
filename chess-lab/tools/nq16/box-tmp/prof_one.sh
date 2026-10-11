#!/bin/bash
# $1 = PHASE_MOE value
E=/mnt/cold-raid6/chess-audit/engine23
ACT=/srv/workspace/flychess/src/Stockfish-act/src/stockfish
OPTS="option.EvalFile=$E/tb.nnue"
i=2
for n in mvr rv2m qvmat n2v2 pd_down pd_up oppb dv_rend dv_QRend dv_qend dv_core nvb op_gambiteer op_acceptor op_even_l0 op_even_l1 op_even_l2p mg_unsafe_king mg_safe_both_same mg_safe_my_castled mg_safe_uncastled mg_safe_other_castled; do
  OPTS="$OPTS option.EvalFile$i=$E/$n.nnue"; i=$((i+1))
done
printf "%s\n" $OPTS | sed "s/option\.EvalFile\([0-9]*\)=/setoption name EvalFile\1 value /" > /tmp/prof_cmd.txt
cat /tmp/prof_cmd.txt
