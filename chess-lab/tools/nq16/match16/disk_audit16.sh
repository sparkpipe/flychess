#!/bin/bash
# Disk audit on one host: what MY chess-era dirs cost, plus total home usage.
echo "== $(hostname) home: $(df -h /home 2>/dev/null | tail -1 | awk '{print $4" free of "$2}')"
du -sh ~/run16_* ~/runanti_* 2>/dev/null
du -sh ~/anti_*.train.bin ~/anti_*.val.bin 2>/dev/null | awk '{a+=$1} END {print "anti bins total listed"}'
ls ~/anti_*.bin ~/[a-z]*.train.bin ~/[a-z]*.val.bin 2>/dev/null | head -40 | xargs -r du -sc 2>/dev/null | tail -1 | awk '{print "bin files: "$1" KB-total"}'
du -sh ~/d20out.tsv ~/d20job.txt 2>/dev/null
echo "tmp helpers: $(ls ~/cur16.py ~/pick16.py ~/curanti.py ~/pickanti.py ~/free_sparks.sh 2>/dev/null | wc -l)"
echo "procs: $(pgrep -f "train\.py|train23|watch_|anti_watch|d20_spark|net_prune" | wc -l)"
crontab -l 2>/dev/null | grep -v "^#" | grep -v "^$" | head -5 || echo "no crontab"
