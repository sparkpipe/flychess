#!/bin/bash
R=/mnt/cold-raid6/chess-audit
NNUE=/srv/workspace/flychess/src/nnue-pytorch
for EXP in qvmat rv2m op_acceptor op_gambiteer op_even_l1 "op_even_l2+" n2v2; do
  D=$R/runs23/anti/$EXP
  mkdir -p $D
  T=$R/train23/anti/$EXP.train.bin
  V=$R/train23/anti/$EXP.val.bin
  N=$(( $(stat -c %s $T) / 40 )); [ $N -lt 1000 ] && N=1000
  cd $NNUE
  setsid nohup python3 train.py $T     --validation-datasets $V --validation-size 30000     --check-val-every-n-epoch 1 --epoch-size $N --batch-size 2048     --max-time 06:00:00:00 --max-epochs 3000 --network-save-period 20 --threads 2     --default-root-dir $D >> $D/box.train.log 2>&1 < /dev/null &
  echo "launched $EXP (epoch-size $N) pid $!"
  sleep 3
done
