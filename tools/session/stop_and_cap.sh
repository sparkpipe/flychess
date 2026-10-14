#!/bin/bash
# Stop queue+trainer, keep piece_down's best, relaunch with 1h/expert caps.
cd /srv/workspace/chess-active
pgrep -f train_continue.sh | xargs -r kill
sleep 1
pgrep -f "train.py" | xargs -r kill
sleep 5
pgrep -f "train.py" | xargs -r kill -9 2>/dev/null
# serialize piece_down best ckpt from the continuation run
CK=$(python3 - <<'EOF'
import glob, re
cks = glob.glob("/srv/workspace/chess-active/runs-continue/piece_down/lightning_logs/version_*/checkpoints/epoch=*.ckpt")
cks += glob.glob("/srv/workspace/chess-active/runs-continue/piece_down/lightning_logs/version_*/checkpoints/last.ckpt")
def ep(f):
    m = re.search(r"epoch=(\d+)", f)
    return int(m.group(1)) if m else 10**9  # last.ckpt = newest
cks.sort(key=ep)
print(cks[-1] if cks else "")
EOF
)
if [ -n "$CK" ]; then
  cd /srv/workspace/flychess/src/nnue-pytorch
  python3 serialize.py "$CK" /srv/workspace/chess-active/runs-continue/piece_down/best.nnue >/dev/null 2>&1 && echo "piece_down best kept ($(basename $CK))"
fi
