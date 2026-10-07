#!/bin/bash
# ROUTE PARITY v2 — python-side diff, legal move sequences
set -e
SF=/srv/workspace/flychess/src/Stockfish-act/src/stockfish
SEGD=/mnt/cold-raid6/chess-audit/wp_fit/segments
TMP=/tmp/parity
mkdir -p $TMP

awk -F'|' 'NR % 9000 == 1 {print $1}' $SEGD/pos_*.clean.tsv | sort -u > $TMP/fens.txt

{
  echo "uci"
  while read -r fen; do
    echo "position fen $fen"
    echo "route"
  done < $TMP/fens.txt
  echo "quit"
} | $SF 2>/dev/null | grep "^route" | sed "s/route slot //" > $TMP/engine.txt

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

python3 - <<PY
eng = [l.split() for l in open("$TMP/engine.txt")]
py = [l.split() for l in open("$TMP/python.txt")]
n = min(len(eng), len(py))
print(f"engine lines: {len(eng)}  python lines: {len(py)}")
bad = [(i, e, p) for i, (e, p) in enumerate(zip(eng, py)) if e != p]
print(f"compared: {n}  mismatches: {len(bad)}")
for i, e, p in bad[:6]:
    print("  line", i + 1, "engine:", e, "python:", p)
    print("  fen:", open("$TMP/fens.txt").readlines()[i].strip()[:80])
PY

echo "--- transition (QGD with Nb8-a6 recapture line) ---"
{
  echo "uci"
  echo "position startpos moves d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 f1d3 d5c4 d3c4 b7b5"
  echo "route"
  echo "position startpos moves d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 f1d3 d5c4 d3c4 b7b5 c4b5 a7a6"
  echo "route"
  echo "position startpos moves d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 f1d3 d5c4 d3c4 b7b5 c4b5 a7a6 b5a6 b8a6"
  echo "route"
  echo "position startpos moves d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 f1d3 d5c4 d3c4 b7b5 c4b5 a7a6 b5a6 b8a6 e1g1 c7c6"
  echo "route"
  echo "quit"
} | $SF 2>/dev/null | grep "^route"
echo "(expect: hold through Bxb5/Nxa6 trades, handoff ~3 plies after last capture)"
