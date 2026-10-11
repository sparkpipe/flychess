#!/bin/bash
# runs ON a spark. arg1 = first shard index. launches 14 d1-12 labelers.
first=$1; s=$(hostname -s)
for k in $(seq 0 13); do
  n=$((first+k)); f=$(printf "r%03d.shard" $n)
  [ -f /home/$s/$f ] || { echo "missing $f"; continue; }
  cat > /tmp/w_$n.sh << WE
#!/bin/bash
while true; do
  python3 -u /home/$s/depth_worker.py /home/$s/$f
  echo "\$(date +%H:%M:%S) w$n exited rc=\$? — restart 10s"
  sleep 10
done
WE
  chmod +x /tmp/w_$n.sh
  setsid nohup timeout 3600 env NODE=pddepth-$s-$n SVC=http://192.168.50.4:8123 SF=/home/$s/sf17_arm bash /tmp/w_$n.sh >> /home/$s/pdgrant.log 2>&1 < /dev/null &
done
sleep 3
echo "$s launched: $(pgrep -f sf17_arm | wc -l) SF procs running"
