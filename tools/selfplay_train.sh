#!/bin/bash
# SELF-PLAY TRAINING LOOP for moe13 (the 1-hour-2400 recipe, MoE version):
#   1. moe13 plays self-play games (current nets)
#   2. every position evaluated by the engine itself (routed expert eval)
#   3. positions tagged with routing slot -> per-expert training bins
#   4. each expert retrains on its positions
#   5. reload and repeat
set -u
R=/mnt/cold-raid6/chess-audit
A=$HOME/arena
SP=$R/selfplay
ITER=${ITER:-0}

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

mkdir -p $SP/iter_$ITER $SP/games

echo "=== SP ITER $ITER: generate games $(date -u) ==="
cd $A
./fastchess-linux-x86-64/fastchess \
  -engine name=moe13a cmd=$A/fork_default.sh $OPTS \
  -engine name=moe13b cmd=$A/fork_default.sh $OPTS \
  -each proto=uci st=0.3 -rounds 50 -repeat \
  -openings file=$R/otb_book.epd format=epd order=random \
  -pgnout file=$SP/games/iter_${ITER}.pgn \
  >> $SP/games/iter_${ITER}.log 2>&1

echo "=== SP ITER $ITER: extract positions + evals $(date -u) ==="
python3 /home/spec/chess-lab/tools/selfplay_extract.py \
  $SP/games/iter_${ITER}.pgn \
  $SP/iter_$ITER/positions.txt

echo "=== SP ITER $ITER: evaluate + classify + pack $(date -u) ==="
# evaluate with the fork itself (same nets, so evals are self-consistent)
python3 /home/spec/chess-lab/tools/selfplay_eval.py \
  $SP/iter_$ITER/positions.txt \
  $SP/iter_$ITER/evals.txt \
  "$OPTS"

python3 /home/spec/chess-lab/tools/selfplay_pack.py \
  $SP/iter_$ITER/evals.txt \
  $SP/iter_$ITER/bins

echo "=== SP ITER $ITER: retrain experts $(date -u) ==="
cd /home/spec/nnue-pytorch
for expert in balanced_l0 balanced_l1 balanced_l2 balanced_l3 nvb nvr bvr \
              rv2m qvmat oppb dvoretsky exchanges; do
  bin=$SP/iter_$ITER/bins/$expert.bin
  [ ! -f "$bin" ] && continue
  n=$(($(stat -c%s "$bin") / 40))
  [ "$n" -lt 5000 ] && continue   # too few positions to train on
  echo "  training $expert ($n positions)"
  python3 train.py "$bin" \
    --validation-size $((n / 10)) \
    --check-val-every-n-epoch 1 \
    --epoch-size "$n" \
    --batch-size 8192 --max-time 00:00:10:00 --max-epochs 200 \
    --resume-from-model "$R/nets3/$expert.nnue" \
    --default-root-dir $SP/iter_$ITER/runs/$expert \
    >> $SP/iter_$ITER/$expert.trainlog 2>&1
  ck=$(ls -t $SP/iter_$ITER/runs/$expert/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
  [ -n "$ck" ] && python3 serialize.py "$ck" "$R/nets3/$expert.nnue" >> $SP/iter_$ITER/$expert.trainlog 2>&1
done

echo "=== SP ITER $ITER COMPLETE $(date -u) ==="
touch $SP/iter_$ITER/DONE
