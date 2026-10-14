#!/bin/bash
# Deploy ONE shard worker to a spark. Usage: spark_deploy_one.sh <sparkN> (on box)
s=$1
cat > /tmp/w3_${s}.sh <<WEOF
#!/bin/bash
while true; do
  python3 ~/dbnet_worker3.py ${s} 18 20 http://192.168.50.4:8123 ~/sf17_arm ~/shard >> ~/d20shard.log 2>&1
  echo "worker exited rc=\$? — restart in 10s" >> ~/d20shard.log
  sleep 10
done
WEOF
scp -q -o BatchMode=yes -o ConnectTimeout=10 /tmp/w3_${s}.sh ~/dbnet_worker3.py ${s}: 2>/dev/null || { echo "${s}: scp FAIL"; exit 1; }
ssh -o BatchMode=yes -o ConnectTimeout=10 ${s} "mv ~/w3_${s}.sh ~/start_worker3.sh 2>/dev/null; chmod +x ~/start_worker3.sh; pgrep -f 'dbnet_worker2.p[y]' | xargs -r kill 2>/dev/null; pgrep -f 'start_worker[2]' | xargs -r kill 2>/dev/null; sleep 1; setsid nohup bash ~/start_worker3.sh > /dev/null 2>&1 < /dev/null & sleep 2; pgrep -f 'dbnet_worker[3]' | wc -l" 2>/dev/null | grep -q "^[1-9]" && echo "${s}: v3 started" || echo "${s}: ssh FAIL"
rm -f /tmp/w3_${s}.sh
