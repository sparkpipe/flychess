#!/bin/bash
# Stop nQ16 MAIN training cleanly on this host: watcher first, then trainer.
# d20 workers (~/d20_spark_worker.py) are untouched — pattern only matches run16 trainers.
for w in $(pgrep -f "train_watch_generic.sh|/tmp/watch_"); do kill "$w" 2>/dev/null; done
sleep 1
for p in $(pgrep -f "train\.py .*run16_"); do kill "$p" 2>/dev/null; done
sleep 4
for p in $(pgrep -f "train\.py .*run16_"); do kill -9 "$p" 2>/dev/null; done
sleep 1
if pgrep -f "train\.py .*run16_" >/dev/null; then
  echo "STILL RUNNING:"; pgrep -fa "train\.py .*run16_"
else
  echo "main trainer stopped on $(hostname)"
fi
