#!/bin/bash
# spark worker wrapper: restart forever, backoff on api failures
while true; do
  python3 ~/dbnet_worker2.py spark0 16 20 http://192.168.50.4:8123 ~/sf17_arm >> ~/d20traj.log 2>&1
  echo "$(date +%H:%M:%S) worker exited rc=$? — restarting in 10s" >> ~/d20traj.log
  sleep 10
done
