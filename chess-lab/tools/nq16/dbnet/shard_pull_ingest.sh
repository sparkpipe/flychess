#!/bin/bash
# Shard puller+ingestor (box): pulls each spark's ~/shard/*.jsonl, feeds to DB ingestor.
# Runs as the single DB writer; sparks never touch the DB.
set -u
SHARD=/srv/workspace/chess-active/shards
mkdir -p $SHARD
SPARKS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
while true; do
  for s in $SPARKS; do
    scp -q -o BatchMode=yes -o ConnectTimeout=8 $s:shard/*.jsonl $SHARD/${s}_pull.jsonl 2>/dev/null
    ssh -o BatchMode=yes -o ConnectTimeout=8 $s 'truncate -s 0 ~/shard/*.jsonl 2>/dev/null' 2>/dev/null
  done
  python3 /srv/workspace/chess-active/shard_ingest_local.py
  sleep 20
done
