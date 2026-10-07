#!/bin/bash
# NET LOAD SMOKE — engine23 all-23 EvalFile loading + eval sanity
SF=/srv/workspace/flychess/src/Stockfish-act/src/stockfish
N=/mnt/cold-raid6/chess-audit/engine23
declare -A MAP=( [tb]=tb [mvr]=mvr [rv2m]=rv2m [qvmat]=qvmat [n2v2]=n2v2 [pd_down]=pd_down [pd_up]=pd_up [oppb]=oppb [dv_rend]=dv_rend [dv_QRend]=dv_QRend [dv_qend]=dv_qend [dv_core]=dv_core [nvb]=nvb [op_gambiteer]=op_gambiteer [op_acceptor]=op_acceptor [op_even_l0]=op_even_l0 [op_even_l1]=op_even_l1 [op_even_l2+]=op_even_l2+ [mg_unsafe_king]=mg_unsafe_king [mg_safe_both_same]=mg_safe_both_same [mg_safe_my_castled]=mg_safe_my_castled [mg_safe_uncastled]=mg_safe_uncastled [mg_safe_other_castled]=mg_safe_other_castled )
# nets in engine23 are named by expert; find each file (pick -last or -e*)
FILES=$(ls $N/*.nnue | head -30)
{
  echo "uci"
  i=0
  for f in $N/tb.nnue $N/mvr*.nnue $N/rv2m*.nnue $N/qvmat*.nnue $N/n2v2*.nnue $N/pd_down*.nnue $N/pd_up*.nnue $N/oppb*.nnue $N/dv_rend*.nnue $N/dv_QRend*.nnue $N/dv_qend*.nnue $N/dv_core*.nnue $N/nvb*.nnue $N/op_gambiteer*.nnue $N/op_acceptor*.nnue $N/op_even_l0*.nnue $N/op_even_l1*.nnue $N/op_even_l2+*.nnue $N/mg_unsafe_king*.nnue $N/mg_safe_both_same*.nnue $N/mg_safe_my_castled*.nnue $N/mg_safe_uncastled*.nnue $N/mg_safe_other_castled*.nnue; do
    [ -f "$f" ] || { echo "MISSING: $f" >&2; continue; }
    if [ $i -eq 0 ]; then echo "setoption name EvalFile value $f"; else echo "setoption name EvalFile$((i+1)) value $f"; fi
    i=$((i+1))
  done
  echo "isready"
  echo "position startpos"
  echo "go depth 10"
  sleep 4
  echo "quit"
} | $SF 2>&1 | tr -d "\000" | grep -aE "final|bestmove|info depth 10 |Error|error|Nets|net" | head -8
echo "---"
echo "loaded count: $i"
