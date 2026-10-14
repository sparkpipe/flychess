#!/bin/bash
# Finish d20 sweep on the rtx5090 (operator: box only, sparks released).
# Uses the dbnet service (HTTP) + worker on box with 12 cores.
cd /srv/workspace/chess-active
pgrep -f "shard_pull_inges[t]" | xargs -r kill 2>/dev/null
sleep 1
# check remaining
python3 -c "
import sqlite3
c=sqlite3.connect('file:$PWD/store/games.db?mode=ro', uri=True)
u=c.execute(\"select count(*) from positions p where not exists(select 1 from labels l where l.fen=p.fen and l.engine='sf17' and l.depth=20)\").fetchone()[0]
print(f'remaining d20: {u:,}')" 2>/dev/null || echo "check failed"
# run box worker (12 cores, spare some for training)
SF17=/srv/workspace/flychess/src/sf17/src/stockfish
nohup python3 dbnet_worker2.py boxfinish 12 20 http://127.0.0.1:8123 $SF17 > d20_box_finish.log 2>&1 < /dev/null &
echo box-d20-finisher-started
