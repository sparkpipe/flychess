#!/bin/bash
# Deploy shard-mode workers (v3) to all 16 sparks + start box ingestor.
# Workers: 18 procs × 1T, append to per-node JSONL on the box via NFS-free
# local path... sparks write locally, box pulls. Simpler: sparks write to
# their own ~/shard.jsonl; box ingestor pulls via ssh every cycle.
# ACTUALLY simplest for now: sparks append over sshfs-free direct — the
# shard dir is on the box and sparks mount nothing; workers write locally,
# box ingestor ssh-cats. To avoid complexity: workers write to local file,
# box pulls periodically.
set -u
SPARKS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
for s in $SPARKS; do
  cat > /tmp/w3_$s.sh <<WEOF
#!/bin/bash
while true; do
  python3 ~/dbnet_worker3.py $s 18 20 http://192.168.50.4:8123 ~/sf17_arm >> ~/d20shard.log 2>&1
  echo "worker exited rc=\$? — restart in 10s" >> ~/d20shard.log
  sleep 10
done
WEOF
  scp -q -o BatchMode=yes -o ConnectTimeout=10 /tmp/w3_$s.sh ~/dbnet_worker3.py $s: 2>/dev/null \
    && ssh -o BatchMode=yes -o ConnectTimeout=10 $s "
      mv ~/w3_$s.sh ~/start_worker3.sh 2>/dev/null; mv ~/dbnet_worker3.py ~/dbnet_worker3.py 2>/dev/null
      chmod +x ~/start_worker3.sh
      pgrep -f 'start_worker[2]' | xargs -r kill 2>/dev/null
      pgrep -f 'dbnet_worker2.py' | xargs -r kill 2>/dev/null
      sleep 1
      setsid nohup bash ~/start_worker3.sh > /dev/null 2>&1 < /dev/null &
      sleep 2
      echo ok" 2>/dev/null \
    && echo "$s: v3 shard worker started" || echo "$s: FAIL"
  rm -f /tmp/w3_$s.sh
done
