#!/bin/bash
# PARITY23 — engine route23 vs python router23 on engine23 corpus positions
set -e
SF=/srv/workspace/flychess/src/Stockfish-act/src/stockfish
NETS=/mnt/cold-raid6/chess-audit/engine23
cd /srv/workspace/flychess/src/chess-lab

# load the 23 nets as EvalFile..EvalFile23 via UCI setoption
{
  echo "uci"
  i=0
  for e in tb mvr rv2m qvmat n2v2 pd_down pd_up oppb dv_rend dv_QRend dv_qend dv_core nvb op_gambiteer op_acceptor op_even_l0 op_even_l1 op_even_l2+ mg_unsafe_king mg_safe_both_same mg_safe_my_castled mg_safe_uncastled mg_safe_other_castled; do
    if [ $i -eq 0 ]; then echo "setoption name EvalFile value $NETS/$e.nnue"; else echo "setoption name EvalFile$((i+1)) value $NETS/$e.nnue"; fi
    i=$((i+1))
  done
  echo "isready"
  # sample fens
  awk -F"|" "NR<=30000 {print \$1}" /mnt/cold-raid6/chess-audit/train23/main/nvb.train.bin 2>/dev/null | head -0  # bins not text; use segments instead
  for fen in $(awk -F"|" "{print \$1}" /mnt/cold-raid6/chess-audit/wp_fit/segments/pos_55to70.clean.tsv | head -300); do
    echo "position fen $fen"
    echo "route"
  done
  echo "quit"
} | $SF 2>/dev/null | grep "^route" | awk "{print \$3}" > /tmp/engine_routes.txt

python3 - <<PY
import sys
sys.path.insert(0, "tools")
import chess, router12 as R
fens = []
for line in open("/mnt/cold-raid6/chess-audit/wp_fit/segments/pos_55to70.clean.tsv", errors="replace"):
    fens.append(line.split("|")[0])
    if len(fens) >= 2000:
        break
py = [R.route23(f) for f in fens]
eng = [l.strip() for l in open("/tmp/engine_routes.txt")]
print(f"positions: {len(fens)} engine-routed: {len(eng)}")
mism = sum(1 for a, b in zip(py, eng) if a != b)
print(f"mismatches: {mism}")
for i, (a, b) in enumerate(zip(py, eng)):
    if a != b:
        print(" ", fens[i][:60], "py:", a, "eng:", b)
    if i > 20 and mism:
        break
PY
