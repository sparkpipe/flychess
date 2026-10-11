#!/bin/bash
# Net retention + backup: per spark — prune nets/ to (best-val epochs + newest 12),
# rsync newest hourly ckpt + best nets to cold RAID backup dir.
R=$1; E=$2
BKP=/mnt/cold-raid6/chess-audit/train16_backup
SSH=spec@192.168.50.4
# best epochs so far from metrics.csv (top-5 val_corr rows)
f=$(ls $R/lightning_logs/version_*/metrics.csv 2>/dev/null | tail -1)
[ -z "$f" ] && exit 0
best=$(awk -F, 'NF>8 && $8!="" {print $1" "$8}' $f | sort -k2 -gr | head -5 | awk '{print $1}')
cd $R/nets 2>/dev/null || exit 0
keep=""
for b in $best; do keep="$keep -o -name ${E}_e$b.nnue"; done
keep=$(echo $keep | sed 's/^-o //')
# newest 12 by epoch number
newest=$(ls | grep -o "_e[0-9]*" | grep -o "[0-9]*" | sort -n | tail -12 | while read n; do echo "-o -name ${E}_e$n.nnue"; done | tr "\n" " ")
newest=$(echo $newest | sed 's/ -o / -o /g; s/^-o //')
prune_count=$(ls | wc -l)
find . -maxdepth 1 -name "*.nnue" ! \( $(echo $keep) \) ! \( $(echo $newest | sed 's/^ *//') \) -delete 2>/dev/null
after=$(ls | wc -l)
echo "$E nets: $prune_count -> $after (kept best+newest)"
# backup: newest hourly ckpt + kept nets (box-side pull via ssh from box cron)
