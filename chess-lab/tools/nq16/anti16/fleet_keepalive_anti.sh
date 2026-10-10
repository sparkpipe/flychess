#!/bin/bash
# Anti16 fleet keepalive: prune nets, backup to cold RAID, supervise watchers, report dead trainers.
# Watchers are auto-restarted; trainers are NEVER auto-restarted (report only).
EXPERTS="tb mvr rv2m qvmat nvb piece_down oppb dv_Q dv_R dv_rest op_pawnimb op_even_l0 op_even_l2p mg_unsafe mg_safe"
SPARKS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkd sparke sparkf"
BKP=/mnt/cold-raid6/chess-audit/anti16_backup
LOG=/tmp/fleet_keepalive.log
mkdir -p $BKP
echo "$(date +%H:%M) anti cycle start" >> $LOG
i=0
for s in $SPARKS; do
  e=$(echo $EXPERTS | cut -d" " -f$((i+1)))
  ssh -o ConnectTimeout=8 -o BatchMode=yes $s "/tmp/net_prune.sh ~/runanti_$e $e" >> $LOG 2>&1
  # watcher supervision (bracket trick: pattern must not match the probing shell itself)
  if ! ssh -o ConnectTimeout=8 -o BatchMode=yes $s "pgrep -f 'anti_watch_spar[k]\.sh .*runanti_$e'" >/dev/null 2>&1; then
    ssh -o ConnectTimeout=8 -o BatchMode=yes $s "nohup bash /tmp/anti_watch_spark.sh ~/runanti_$e $e 3600 9999999 >> ~/runanti_$e/watch_nohup.log 2>&1 </dev/null &" >> $LOG 2>&1
    echo "$s: watcher restarted for $e" >> $LOG
  fi
  # trainer status (report only)
  if ! ssh -o ConnectTimeout=8 -o BatchMode=yes $s "pgrep -f 'train\.py .*runanti_$e'" >/dev/null 2>&1; then
    echo "$(date +%H:%M) $s: TRAINER DEAD for $e" >> $LOG
  fi
  # backup: newest hourly ckpt (throttled) + nets
  mkdir -p $BKP/$e
  ck=$(ssh -o ConnectTimeout=8 -o BatchMode=yes $s "ls -t ~/runanti_$e/checkpoint.*.ckpt 2>/dev/null | head -1" 2>/dev/null)
  if [ -n "$ck" ]; then
    base=$(basename "$ck")
    if [ ! -f $BKP/$e/$base ]; then
      scp -q -o ConnectTimeout=15 -o BatchMode=yes $s:"$ck" $BKP/$e/$base >> $LOG 2>&1
      ls -t $BKP/$e/checkpoint.* 2>/dev/null | tail -n +3 | xargs -r rm -f
    fi
  fi
  rsync -q --delete -e "ssh -o ConnectTimeout=10 -o BatchMode=yes" $s:runanti_$e/nets/ $BKP/$e/nets/ >> $LOG 2>&1
  i=$((i+1))
done
# box run: op_even_l1
BR=/srv/workspace/chess-active/runanti_op_even_l1
if ! pgrep -f "train_watch_generic.sh $BR" >/dev/null 2>&1; then
  nohup bash /tmp/train_watch_generic.sh $BR op_even_l1 3600 9999999 >> $BR/watch_nohup.log 2>&1 </dev/null &
  echo "$(date +%H:%M) box: watcher restarted op_even_l1" >> $LOG
fi
pgrep -f "train\.py .*runanti_op_even_l1" >/dev/null 2>&1 || echo "$(date +%H:%M) box: TRAINER DEAD op_even_l1" >> $LOG
mkdir -p $BKP/op_even_l1
ck=$(ls -t $BR/checkpoint.*.ckpt 2>/dev/null | head -1)
if [ -n "$ck" ] && [ ! -f $BKP/op_even_l1/$(basename "$ck") ]; then
  cp "$ck" $BKP/op_even_l1/$(basename "$ck")
  ls -t $BKP/op_even_l1/checkpoint.* 2>/dev/null | tail -n +3 | xargs -r rm -f
fi
rsync -q --delete $BR/nets/ $BKP/op_even_l1/nets/ >> $LOG 2>&1
/tmp/net_prune.sh $BR op_even_l1 >> $LOG 2>&1
echo "$(date +%H:%M) anti cycle done" >> $LOG
