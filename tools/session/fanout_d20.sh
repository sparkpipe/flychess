#!/bin/bash
# Fleet-wide d20 trajectory sweep: 18 procs × 1 thread per spark, auto-restart wrapper.
# Usage: bash fanout_d20.sh   (runs on box, fans out to all 15 sparks)
set -u
SPARKS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
for s in $SPARKS; do
  echo "$s" | grep -q "^sparkc$" && { echo "$s: SKIP (serving)"; continue; }
  cat > /tmp/w_$s.sh <<WEOF
#!/bin/bash
while true; do
  python3 ~/dbnet_worker2.py $s 18 20 http://192.168.50.4:8123 ~/sf17_arm >> ~/d20traj.log 2>&1
  echo "worker exited rc=\$? — restart in 10s" >> ~/d20traj.log
  sleep 10
done
WEOF
  scp -q -o BatchMode=yes -o ConnectTimeout=10 /tmp/w_$s.sh $s:~/start_worker2.sh 2>/dev/null \
    && scp -q -o BatchMode=yes -o ConnectTimeout=10 ~/dbnet_worker2.py $s: 2>/dev/null \
    && ssh -o BatchMode=yes -o ConnectTimeout=10 $s "chmod +x ~/start_worker2.sh; pgrep -f 'start_worker[2]' | xargs -r kill 2>/dev/null; pgrep -f 'dbnet_worker[2]' | xargs -r kill 2>/dev/null; sleep 1; rm -f ~/d20traj.log; nohup bash ~/start_worker2.sh > /dev/null 2>&1 < /dev/null &" 2>/dev/null \
    && echo "$s: 18-proc trajectory worker started" || echo "$s: FAIL"
  rm -f /tmp/w_$s.sh
done
