#!/bin/bash
# REBUILD CHAIN: rows -> assemble16 (all stages) -> marker -> night launcher2
LOG=/tmp/rebuild_chain.log
log(){ echo "$(date +%H:%M) $1" >> $LOG; }
while pgrep -f "miniature_rows2.py" > /dev/null; do sleep 60; done
log "rows done: $(wc -l < /srv/workspace/chess-active/miniature_rows.tsv)"
cd /srv/workspace/flychess/src/nnue-pytorch
python3 /tmp/assemble16r.py >> /tmp/assemble16r.log 2>&1
log "assemble done"
touch /srv/workspace/chess-active/.assemble_done
chmod +x /tmp/night_launcher2.sh
/tmp/night_launcher2.sh
log "night launch complete"
