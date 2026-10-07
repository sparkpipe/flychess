#!/bin/bash
cd /srv/workspace/chess-active
pgrep -f dbnet_ingest_minis | xargs -r kill
sleep 1
python3 drop_positions.py
nohup python3 dbnet_ingest_minis.py > ingest_minis2.log 2>&1 < /dev/null &
echo reingest-running
