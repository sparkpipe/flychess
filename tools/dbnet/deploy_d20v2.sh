#!/bin/bash
# deploy one 16-core d20 worker on a spark (cursor protocol, port 8125)
s=$1
scp -q -o BatchMode=yes -o ConnectTimeout=10 /srv/workspace/chess-active/d20_worker.py $s: || { echo "$s: scp FAIL"; exit 1; }
ssh -o BatchMode=yes -o ConnectTimeout=10 $s "pkill -f \"d20_[w]orker.py\" 2>/dev/null; true"
sleep 1
ssh -o BatchMode=yes -o ConnectTimeout=10 $s "cat > /home/$s/d20v2_wrap.sh << W
#!/bin/bash
while true; do
  python3 -u /home/$s/d20_worker.py $s 16 http://192.168.50.4:8125 /home/$s/sf17_arm >> /home/$s/d20v2.log 2>&1
  rc=\\\$?
  [ \\\$rc -eq 0 ] && exit 0
  sleep 10
done
W
chmod +x /home/$s/d20v2_wrap.sh
setsid nohup bash /home/$s/d20v2_wrap.sh > /dev/null 2>&1 < /dev/null &
sleep 2
echo \"$s: worker=\$(pgrep -f \\\"python3 -u /home/$s/d20_worker\\\" | head -1) sf=\$(ps -eo comm | grep -c sf17_arm)\""
