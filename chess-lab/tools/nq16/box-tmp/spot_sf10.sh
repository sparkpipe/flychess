#!/bin/bash
# SPOT TEST — nQ.23 vs SF8 and vs nQ, at 1s and 10s/move.
# Proven fastchess pattern from tourney7.sh. 6 games per pairing (3 rounds x repeat).
set -u
FC=/srv/workspace/flychess/src/chess-lab/fastchess-linux-x86-64/fastchess
ACT=/srv/workspace/flychess/src/Stockfish-act/src/stockfish   # 23-slot nQ.23
FORK=/srv/workspace/flychess/src/Stockfish/src/stockfish     # 13-slot nQ
SF8=/usr/games/stockfish
E=/mnt/cold-raid6/chess-audit/engine23
N=/mnt/cold-raid6/chess-audit/nets
R=/mnt/cold-raid6/chess-audit
OUT=$R/spot23
mkdir -p $OUT

OPTS23="option.EvalFile=$E/tb.nnue option.EvalFile2=$E/mvr.nnue option.EvalFile3=$E/rv2m.nnue option.EvalFile4=$E/qvmat.nnue option.EvalFile5=$E/n2v2.nnue option.EvalFile6=$E/pd_down.nnue option.EvalFile7=$E/pd_up.nnue option.EvalFile8=$E/oppb.nnue option.EvalFile9=$E/dv_rend.nnue option.EvalFile10=$E/dv_QRend.nnue option.EvalFile11=$E/dv_qend.nnue option.EvalFile12=$E/dv_core.nnue option.EvalFile13=$E/nvb.nnue option.EvalFile14=$E/op_gambiteer.nnue option.EvalFile15=$E/op_acceptor.nnue option.EvalFile16=$E/op_even_l0.nnue option.EvalFile17=$E/op_even_l1.nnue option.EvalFile18=$E/op_even_l2p.nnue option.EvalFile19=$E/mg_unsafe_king.nnue option.EvalFile20=$E/mg_safe_both_same.nnue option.EvalFile21=$E/mg_safe_my_castled.nnue option.EvalFile22=$E/mg_safe_uncastled.nnue option.EvalFile23=$E/mg_safe_other_castled.nnue"

OPTSNQ="option.EvalFile=$N/balanced_l0.nnue option.EvalFile2=$N/balanced_l1.nnue option.EvalFile3=$N/balanced_l2.nnue option.EvalFile4=$N/balanced_l3.nnue option.EvalFile5=$N/nvb.nnue option.EvalFile6=$N/nvr.nnue option.EvalFile7=$N/bvr.nnue option.EvalFile8=$N/rv2m.nnue option.EvalFile9=$N/qvmat.nnue option.EvalFile10=$N/oppb.nnue option.EvalFile11=$N/dvoretsky.nnue option.EvalFile12=$N/exchanges.nnue option.EvalFile13=$N/tactics.nnue"

pairing () {  # $1=opp name $2=st $3=opp cmd $4...=opp opts
  local NAME=$1; shift
  local ST=$1; shift
  local BCMD=$1; shift
  local P=$OUT/nQ23_vs_${NAME}_${ST}s
  if [ -f "$P.pgn" ] && [ "$(grep -c '\[Result' $P.pgn 2>/dev/null)" -ge 6 ]; then
    echo "$(date -u +%H:%M) skip $NAME $ST (done)" >> $OUT/progress.log; return
  fi
  echo "$(date -u +%H:%M) start nQ23 vs $NAME @${ST}s" >> $OUT/progress.log
  $FC -engine name=nQ23 cmd=$ACT $OPTS23 \
      -engine name=$NAME cmd=$BCMD "$@" \
      -each proto=uci st=$ST timemargin=200 -rounds 3 -repeat \
      -openings file=$R/3open.epd format=epd \
      -pgnout file=$P.pgn > $P.log 2>&1
  echo "$(date -u +%H:%M) done nQ23 vs $NAME @${ST}s results: $(grep -a '\[Result' $P.pgn 2>/dev/null | sort | uniq -c | tr '\n' ' ')" >> $OUT/progress.log
}

# fast first: 1s pairings, then 10s
pairing SF8 10 $SF8
echo "$(date -u +%H:%M) ALL SPOT PAIRINGS COMPLETE" >> $OUT/progress.log
