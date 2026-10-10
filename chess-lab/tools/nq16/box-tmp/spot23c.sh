#!/bin/bash
# SPOT TEST v3 — correct fastchess syntax
set -u
FC=/srv/workspace/flychess/src/chess-lab/fastchess-linux-x86-64/fastchess
ACT=/srv/workspace/flychess/src/Stockfish-act/src/stockfish
SF8=/usr/games/stockfish
E=/mnt/cold-raid6/chess-audit/engine23
BOOK=/mnt/cold-raid6/chess-audit/3open.epd
OUT=/mnt/cold-raid6/chess-audit/spot23
mkdir -p $OUT

OPTS23="option.EvalFile=$E/tb.nnue option.EvalFile2=$E/mvr.nnue option.EvalFile3=$E/rv2m.nnue option.EvalFile4=$E/qvmat.nnue option.EvalFile5=$E/n2v2.nnue option.EvalFile6=$E/pd_down.nnue option.EvalFile7=$E/pd_up.nnue option.EvalFile8=$E/oppb.nnue option.EvalFile9=$E/dv_rend.nnue option.EvalFile10=$E/dv_QRend.nnue option.EvalFile11=$E/dv_qend.nnue option.EvalFile12=$E/dv_core.nnue option.EvalFile13=$E/nvb.nnue option.EvalFile14=$E/op_gambiteer.nnue option.EvalFile15=$E/op_acceptor.nnue option.EvalFile16=$E/op_even_l0.nnue option.EvalFile17=$E/op_even_l1.nnue option.EvalFile18=$E/op_even_l2p.nnue option.EvalFile19=$E/mg_unsafe_king.nnue option.EvalFile20=$E/mg_safe_both_same.nnue option.EvalFile21=$E/mg_safe_my_castled.nnue option.EvalFile22=$E/mg_safe_uncastled.nnue option.EvalFile23=$E/mg_safe_other_castled.nnue"

ST=${1:-1}
ROUNDS=${2:-3}

echo "=== nQ.23 vs SF8 @ ${ST}s/move, ${ROUNDS} rounds (opening pairs) $(date -u +%H:%M) ==="
$FC \
  -each proto=uci tc=none st=$ST timemargin=200 \
  -concurrency 1 -rounds $ROUNDS -repeat 2 \
  -openings file=$BOOK format=epd order=sequential \
  -engine name=nQ23 cmd=$ACT $OPTS23 \
  -engine name=SF8 cmd=$SF8 \
  -pgnout file=$OUT/vsSF8_${ST}s.pgn 2>&1 | tail -4
echo "=== results @${ST}s ==="
grep -a '\[Result' $OUT/vsSF8_${ST}s.pgn | sort | uniq -c
echo "forfeits: $(grep -aci 'forfeit\|adjudicat' $OUT/vsSF8_${ST}s.pgn || echo 0)"
