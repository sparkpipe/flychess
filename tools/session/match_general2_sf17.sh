#!/bin/bash
# general2 vs vanilla SF17 (1s then 10s), runs after the current match set.
cd /srv/workspace/chess-active
# wait for the current match set
while ! grep -q "ALL-GENERAL2-MATCHES-DONE" match_general2.log 2>/dev/null; do
  sleep 60
done
echo "[$(date +%H:%M)] current matches done — starting SF17 matches"
FC=/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess
N16=/srv/workspace/flychess/src/Stockfish/src/stockfish
SF17=/srv/workspace/flychess/src/sf17/src/stockfish
GEN2=/srv/workspace/chess-active/runs-general2/best.nnue
OUT=/srv/workspace/chess-active/matches
G2F="option.EvalFile=$GEN2"

for ST in 1 10; do
  rm -f $OUT/general2_vs_SF17_${ST}s.pgn
  $FC \
    -engine name=general2 cmd=$N16 st=$ST $G2F \
    -engine name=SF17 cmd=$SF17 st=$ST \
    -each proto=uci timemargin=200 \
    -rounds 3 -repeat -recover \
    -openings file=/mnt/cold-raid6/chess-audit/3open.epd format=epd \
    -pgnout file=$OUT/general2_vs_SF17_${ST}s.pgn \
    2>&1 | tee $OUT/general2_vs_SF17_${ST}s.log
  echo "=== general2 vs SF17 ${ST}s: $(grep Points $OUT/general2_vs_SF17_${ST}s.log | tail -1) ==="
done
echo ALL-SF17-MATCHES-DONE
