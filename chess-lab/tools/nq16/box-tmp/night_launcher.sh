#!/bin/bash
# NIGHT LAUNCHER — fires when the miniature pipeline completes.
# 1. waits for collect_chain.sh (labels merged, assemble16 stage 4 done)
# 2. stops ALL test trainings (box + sparks)
# 3. deploys final bins to all 16 sparks (sequential scp — no parallel races)
# 4. ensures data_loader built per spark
# 5. launches 16 expert trainings (one per spark GPU), skip=3, no time limit
# 6. per-epoch .nnue export+delete + hourly wallclock checkpoints per spark
LOG=/tmp/night_launcher.log
log(){ echo "$(date +%H:%M) $1" >> $LOG; }
BOX=spec@192.168.50.4
SPARKS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
EXPERTS="tb mvr rv2m qvmat nvb piece_down oppb dv_Q dv_R dv_rest op_pawnimb op_even_l0 op_even_l1 op_even_l2p mg_unsafe mg_safe"

# 1) wait for the collector chain to finish (marker: its last step writes BUILD COMPLETE)
while ! grep -q "BUILD COMPLETE" /tmp/collect_chain.log 2>/dev/null; do
  sleep 120
done
log "collector finished — verifying final bins"
for e in ; do
  sz=0
  log "bin : 0 train records"
  [ "" -lt 40000 ] && { log "BIN  TOO SMALL — ABORTING NIGHT LAUNCH"; exit 1; }
done
log "final bins ready"

# 2) stop test trainings
ssh -o ConnectTimeout=15 $BOX 'for p in $(ps -eo pid,cmd | grep "[t]rain.py" | awk "{print \$1}"); do kill $p; done; echo box_tests_stopped'
for s in $SPARKS; do
  ssh -o ConnectTimeout=8 -o BatchMode=yes $s 'for p in $(pgrep -f "train.py"); do kill $p; done' 2>/dev/null
done
log "test trainings stopped"

# 3+4+5) deploy and launch per spark
i=0
for s in $SPARKS; do
  e=$(echo $EXPERTS | cut -d" " -f$((i+1)))
  TR=$(stat -c %s /extnvme/active/train16/$e.train.bin 2>/dev/null || echo 1600000)
  N=$((TR / 40)); [ $N -lt 1000 ] && N=1000
  scp -q -o ConnectTimeout=15 -o BatchMode=yes /extnvme/active/train16/$e.train.bin /extnvme/active/train16/$e.val.bin $s:~/
  ssh -o ConnectTimeout=15 -o BatchMode=yes $s "
    cd ~/nnue-pytorch
    python3 -c 'import sys; sys.path.insert(0,\".\"); import data_loader' 2>/dev/null || bash compile_data_loader.sh > /tmp/lb_$e.log 2>&1
    mkdir -p ~/run16_$e
    setsid nohup python3 train.py ~/$e.train.bin \
      --validation-datasets ~/$e.val.bin --validation-size 30000 \
      --check-val-every-n-epoch 1 --epoch-size $N --batch-size 4096 \
      --max-time 99:00:00:00 --max-epochs 100000 --network-save-period 1 \
      --random-fen-skipping 3 \
      --default-root-dir ~/run16_$e > ~/run16_$e/train.log 2>&1 < /dev/null &
    sleep 2; echo $e: \$(pgrep -c train.py) procs"
  log "$s <- $e launched (epoch-size $N)"
  i=$((i+1))
done
log "ALL 16 LAUNCHED"
