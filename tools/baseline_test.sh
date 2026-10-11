#!/bin/bash
# BASELINE: after ROUND3_DONE, load all 13 experts into the fork
# (hard routing, no head) and run fastchess strength tests vs SF 17.1.
R=/mnt/cold-raid6/chess-audit
A=$HOME/arena
while [ ! -f "$R/ROUND3_DONE" ]; do sleep 300; done
echo "=== BASELINE armed at $(date -u) ==="

# slot map per phase_moe.h (dense 0..12, VERIFIED against the header):
# 0=bal0 1=bal1 2=bal2 3=bal3 4=nvb 5=nvr 6=bvr 7=rv2m 8=qvmat 9=oppb
# 10=dvoretsky 11=exchanges 12=tb
OPTS="option.EvalFile=$R/nets3/balanced_l0.nnue"
OPTS="$OPTS option.EvalFile2=$R/nets3/balanced_l1.nnue"
OPTS="$OPTS option.EvalFile3=$R/nets3/balanced_l2.nnue"
OPTS="$OPTS option.EvalFile4=$R/nets3/balanced_l3.nnue"
OPTS="$OPTS option.EvalFile5=$R/nets3/nvb.nnue"
OPTS="$OPTS option.EvalFile6=$R/nets3/nvr.nnue"
OPTS="$OPTS option.EvalFile7=$R/nets3/bvr.nnue"
OPTS="$OPTS option.EvalFile8=$R/nets3/rv2m.nnue"
OPTS="$OPTS option.EvalFile9=$R/nets3/qvmat.nnue"
OPTS="$OPTS option.EvalFile10=$R/nets3/oppb.nnue"
OPTS="$OPTS option.EvalFile11=$R/nets3/dvoretsky.nnue"
OPTS="$OPTS option.EvalFile12=$R/nets3/exchanges.nnue"
OPTS="$OPTS option.EvalFile13=$R/nets2/tb_training.nnue"

cd $A
for SKILL in 2000 2500 2850; do
  ./fastchess-linux-x86-64/fastchess \
    -engine name=moe13 cmd=$A/fork_default.sh $OPTS \
    -engine name=sf17 cmd=/usr/games/stockfish option.UCI_LimitStrength=true option.UCI_Elo=$SKILL \
    -each proto=uci tc=2+0.02 -rounds 20 -repeat \
    -openings file=$R/otb_book.epd format=epd \
    >> $R/baseline_results.txt 2>&1 \
    || echo "UCI_Elo $SKILL FAILED" >> $R/baseline_results.txt
  echo "=== UCI_Elo $SKILL done $(date -u) ===" >> $R/baseline_results.txt
done
touch "$R/BASELINE_DONE"
