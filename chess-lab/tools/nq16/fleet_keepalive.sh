#!/bin/bash
# Fleet keepalive: prune nets (30min) + backup newest hourly ckpt & nets to cold RAID (hourly).
EXPERTS="tb mvr rv2m qvmat nvb piece_down oppb dv_Q dv_R dv_rest op_pawnimb op_even_l0 op_even_l2p mg_unsafe mg_safe"
SPARKS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkd sparke sparkf"
BKP=/mnt/cold-raid6/chess-audit/train16_backup
LOG=/tmp/fleet_keepalive.log
mkdir -p $BKP
echo "$(date +%H:%M) cycle start" >> $LOG
i=0
for s in $SPARKS; do
  e=$(echo $EXPERTS | cut -d" " -f$((i+1)))
  # prune on spark (best-val + newest 12 nets)
  ssh -o ConnectTimeout=8 -o BatchMode=yes $s "/tmp/net_prune.sh ~/run16_$e $e" >> $LOG 2>&1
  # backup: newest hourly ckpt + nets dir (throttled: ckpt only if changed)
  mkdir -p $BKP/$e
  ck=$(ssh -o ConnectTimeout=8 -o BatchMode=yes $s "ls -t ~/run16_$e/checkpoint.*.ckpt 2>/dev/null | head -1" 2>/dev/null)
  if [ -n "$ck" ]; then
    base=$(basename "$ck")
    if [ ! -f $BKP/$e/$base ]; then
      scp -q -o ConnectTimeout=15 -o BatchMode=yes $s:"$ck" $BKP/$e/$base >> $LOG 2>&1
      # keep only 2 newest backups per expert
      ls -t $BKP/$e/checkpoint.* 2>/dev/null | tail -n +3 | xargs -r rm -f
    fi
  fi
  rsync -q --delete -e "ssh -o ConnectTimeout=10 -o BatchMode=yes" $s:run16_$e/nets/ $BKP/$e/nets/ >> $LOG 2>&1
  i=$((i+1))
done
echo "$(date +%H:%M) cycle done" >> $LOG
