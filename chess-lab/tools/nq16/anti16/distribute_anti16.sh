#!/bin/bash
# Distribute anti16 bins + launcher + watcher + fixed nnue.py to all sparks. Runs ON box.
set -u
A=/srv/workspace/chess-active/anti16
NNUEFIX=/srv/workspace/flychess/src/nnue-pytorch/model/nnue.py
MAP="spark0:tb spark1:mvr spark2:rv2m spark3:qvmat spark4:nvb spark5:piece_down \
spark6:oppb spark7:dv_Q spark8:dv_R spark9:dv_rest sparka:op_pawnimb \
sparkb:op_even_l0 sparkd:op_even_l2p sparke:mg_unsafe sparkf:mg_safe"
for pair in $MAP; do
  s=${pair%%:*}; e=${pair##*:}
  (
    scp -q -o ConnectTimeout=10 -o BatchMode=yes $A/$e.train.bin $s:anti_$e.train.bin 2>/dev/null || { echo "$s: BIN FAIL train"; exit 1; }
    scp -q -o ConnectTimeout=10 -o BatchMode=yes $A/$e.val.bin $s:anti_$e.val.bin 2>/dev/null || { echo "$s: BIN FAIL val"; exit 1; }
    scp -q -o ConnectTimeout=10 -o BatchMode=yes /tmp/anti_watch_spark.sh /tmp/launch_anti.sh $s:/tmp/ 2>/dev/null || { echo "$s: SCRIPT FAIL"; exit 1; }
    ssh -o ConnectTimeout=10 -o BatchMode=yes $s "cp -n ~/nnue-pytorch/model/nnue.py ~/nnue-pytorch/model/nnue.py.bak_main16" 2>/dev/null
    scp -q -o ConnectTimeout=10 -o BatchMode=yes $NNUEFIX $s:~/nnue-pytorch/model/nnue.py 2>/dev/null || { echo "$s: NNUE FAIL"; exit 1; }
    echo "$s ($e): deployed"
  ) &
done
wait
echo ALL DEPLOYED
