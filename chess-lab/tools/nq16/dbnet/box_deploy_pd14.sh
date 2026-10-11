#!/bin/bash
# usage: deploy_pd14.sh <sparkN> <first-shard-idx>   (14 shards each)
s=$1; first=$2
D=/srv/workspace/chess-active/pd-depthexp
SH=""; for k in $(seq 0 13); do SH="$SH $D/r$(printf %03d $((first+k))).shard"; done
scp -q -o BatchMode=yes -o ConnectTimeout=10 $D/depth_worker.py $D/pd_launch14.sh $SH $s: || { echo "$s: scp FAIL"; exit 1; }
ssh -o BatchMode=yes -o ConnectTimeout=10 $s "pkill -f \"depth_[w]orker\" 2>/dev/null; pkill -f \"sf17_[a]rm\" 2>/dev/null; true"
sleep 1
ssh -o BatchMode=yes -o ConnectTimeout=10 $s "bash pd_launch14.sh $first"
