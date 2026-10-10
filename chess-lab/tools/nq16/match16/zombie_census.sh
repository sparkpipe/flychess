#!/bin/bash
# Definitive zombie census on one host: any process that could train/relaunch chess work.
hits=$(pgrep -fa "train\.py|train23_queue|train2[3]/|d20_spark_worker|train_watch|anti_watch|night_launch|launch_fleet|spark_train\.|fleet_keepalive|net_prune|backlog_worker|sparkworker|chain_watcher|mini_label_daemon" 2>/dev/null | grep -v "zombie_census" | grep -v "pgrep -fa")
if [ -n "$hits" ]; then
  echo "$(hostname): ZOMBIES FOUND:"
  echo "$hits"
else
  echo "$(hostname): clean (no trainers, watchers, queues, or launchers)"
fi
echo "  crontab: $(crontab -l 2>/dev/null | grep -vc '^#' || echo 0) active entries"
echo "  disk free: $(df -h /home 2>/dev/null | tail -1 | awk '{print $4}')"
