#!/bin/bash
cd /srv/workspace/flychess/src/nnue-pytorch
for e in op_even_l1 op_even_l2p; do
  CK=$(python3 - <<EOF
import glob, re
cks = glob.glob("/srv/workspace/chess-active/runs-fleet1h/$e/lightning_logs/version_*/checkpoints/epoch=*.ckpt")
def ep(f):
    m = re.search(r"epoch=(\d+)", f)
    return int(m.group(1)) if m else -1
cks.sort(key=ep)
print(cks[-1] if cks else "")
EOF
)
  if [ -n "$CK" ]; then
    python3 serialize.py "$CK" /srv/workspace/chess-active/runs-fleet1h/$e/best.nnue >/dev/null 2>&1 \
      && echo "$e best.nnue <- $(basename "$CK")"
  else
    echo "$e: no ckpt"
  fi
done
