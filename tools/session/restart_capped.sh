#!/bin/bash
# Clean restart of the capped continuation queue.
cd /srv/workspace/chess-active
echo "before: queues=$(pgrep -fc 'train_continue[.]sh' || echo 0) trainers=$(pgrep -fc 'trai[n].py' || echo 0)"
pgrep -f "train_continue[.]sh" | xargs -r kill
pgrep -f "trai[n].py" | xargs -r kill
sleep 6
pgrep -f "trai[n].py" | xargs -r kill -9
sleep 1
nohup bash train_continue.sh >> runs-continue/progress.log 2>&1 < /dev/null &
sleep 25
echo "after: queues=$(pgrep -fc 'train_continue[.]sh' || echo 0) trainers=$(pgrep -fc 'trai[n].py' || echo 0)"
tail -2 runs-continue/progress.log
ls -t runs-continue/ | head -3
