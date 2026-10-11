#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$HOME/.local/lib/python3.12/site-packages/nvidia/*/lib:$LD_LIBRARY_PATH
E=$1
cd ~/flychess/nnue-pytorch
python3 train.py ~/flychess/data/$E.bin \
  --max-epochs 200 \
  --batch-size 8192 \
  --validation-size 30000 \
  --check-val-every-n-epoch 1 \
  --default-root-dir ~/flychess/runs/$E \
  --gpus 0 \
  > ~/flychess/logs_$E.log 2>&1
echo "DONE" >> ~/flychess/logs_$E.log
