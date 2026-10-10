#!/bin/bash
# ROUTE PARITY — engine C++ router vs router12.py static, on corpus positions.
set -e
SF=/srv/workspace/flychess/src/Stockfish-act/src/stockfish
SEGD=/mnt/cold-raid6/chess-audit/wp_fit/segments
TMP=/tmp/parity
mkdir -p $TMP

# sample: every 9000th line across all band files (stable, deterministic)
awk 'NR % 9000 == 1 {print $1}' $SEGD/pos_*.clean.tsv | sort -u | head -3000 > $TMP/fens.txt
WC=$(wc -l < $TMP/fens.txt)

# engine side
{
  echo "uci"
  while read -r fen; do
    echo "position fen $fen"
    echo "route"
  done < $TMP/fens.txt
  echo "quit"
} | $SF 2>/dev/null | grep "^route" | awk '{print $3, $4}' > $TMP/engine.txt

# python side (router12 static, same fens)
cd /srv/workspace/flychess/src/chess-lab
python3 - <<PY > $TMP/python.txt
import chess, sys
sys.path.insert(0, "tools")
import router12
for line in open("$TMP/fens.txt"):
    fen = line.strip()
    if not fen:
        continue
    b = chess.Board(fen)
    e = router12.route_static(b) or router12._lock_balanced(b)
    print(router12.EXPERTS12.index(e), e)
PY

paste $TMP/engine.txt $TMP/python.txt > $TMP/joined.txt
N=$(wc -l < $TMP/joined.txt)
MISMATCH=$(awk "{split(\$0,a,\" \"); if (\$1 != \$3 || \$2 != \$4) print}" $TMP/joined.txt | wc -l)
echo "positions: $N   mismatches: $MISMATCH"
awk "{split(\$0,a,\" \"); if (\$1 != \$3 || \$2 != \$4) print}" $TMP/joined.txt | head -5

# transition hold: scripted sequences — engine chain vs corpus rule
echo "--- transition sequences ---"
{
  echo "uci"
  echo "position startpos moves d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 f1d3 d5c4 d3c4 b7b5"
  echo "route"
  echo "position startpos moves d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 f1d3 d5c4 d3c4 b7b5 c4b5 a7a6"
  echo "route"
  echo "position startpos moves d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 f1d3 d5c4 d3c4 b7b5 c4b5 a7a6 b5a6 f8a6"
  echo "route"
  echo "position startpos moves d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 f1d3 d5c4 d3c4 b7b5 c4b5 a7a6 b5a6 f8a6 a6a8 d8a8"
  echo "route"
  echo "quit"
} | $SF 2>/dev/null | grep "^route"
