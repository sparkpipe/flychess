#!/bin/bash
# Start the d20 trajectory worker wrapper on this spark (survives ssh exit).
pkill -f "start_worker2.sh" 2>/dev/null
pkill -f "dbnet_worker2.py" 2>/dev/null
sleep 1
setsid nohup bash ~/start_worker2.sh > /dev/null 2>&1 < /dev/null &
sleep 3
echo "workers: $(pgrep -f dbnet_worker2.py | wc -l)"
