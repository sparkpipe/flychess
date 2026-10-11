#!/bin/bash
# Collects farm-labeled miniature chunks, handles extraction tail, finishes the build.
BOX=spec@192.168.50.4
SPARKS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
LOG=/tmp/collect_chain.log
log(){ echo "$(date +%H:%M) $1" >> $LOG; }

# 1) wait for all 64 chunk labelers to complete
while true; do
  done_ct=0
  i=0
  for s in $SPARKS; do
    for k in 0 1 2 3; do
      c=$(printf "mini_chunk_%02d" $((i*4+k)))
      if ssh -o ConnectTimeout=8 -o BatchMode=yes $s "grep -q COMPLETE ~/$c.log 2>/dev/null"; then
        done_ct=$((done_ct+1))
      fi
    done
    i=$((i+1))
  done
  log "chunks complete: $done_ct/64"
  [ "$done_ct" -eq 64 ] && break
  sleep 120
done

# 2) collect all outputs to the box
log "collecting outputs"
for s in $SPARKS; do
  scp -q -o ConnectTimeout=8 -o BatchMode=yes $s:"~/mini_chunk_*.out" /tmp/mini_out/ 2>/dev/null &
done
wait
cat /tmp/mini_out/*.out > /extnvme/active/miniature_labels.tsv
log "labels pass1: $(wc -l < /extnvme/active/miniature_labels.tsv)"

# 3) extraction tail: label rows added after the snapshot
if ssh -o ConnectTimeout=10 $BOX "pgrep -f '/tmp/miniature_rows.py' >/dev/null"; then
  log "waiting for extraction tail"
  ssh -o ConnectTimeout=10 $BOX "while pgrep -f '/tmp/miniature_rows.py' >/dev/null; do sleep 60; done"
fi
log "snapshot diff for tail rows"
ssh -o ConnectTimeout=10 $BOX "python3 - <<'EOF'
seen = set()
for line in open('/extnvme/active/miniature_labels.tsv'):
    seen.add(line.split('\t',1)[0])
out = open('/tmp/mini_tail.tsv','w')
n = 0
for line in open('/extnvme/active/miniature_rows.tsv'):
    fen = line.split('\t')[0]
    if fen not in seen:
        out.write(line); n += 1
out.close()
print('tail rows:', n)
EOF"
TAILN=$(ssh -o ConnectTimeout=10 $BOX "wc -l < /tmp/mini_tail.tsv")
if [ "$TAILN" -gt 1000 ]; then
  log "labeling tail: $TAILN rows"
  ssh -o ConnectTimeout=10 $BOX "split -l 200000 -d /tmp/mini_tail.tsv /tmp/mini_tail_ck_"
  i=0
  for f in $(ssh -o ConnectTimeout=10 $BOX "ls /tmp/mini_tail_ck_*"); do
    s=$(echo $SPARKS | cut -d" " -f$(( (i % 16) + 1 )))
    ssh -o ConnectTimeout=10 $BOX "scp -q -o ConnectTimeout=8 $f $s:~/$f"
    ssh -o ConnectTimeout=8 -o BatchMode=yes $s "setsid nohup python3 ~/spark_labeler.py ~/$f 16 > ~/$f.log 2>&1 < /dev/null &"
    i=$((i+1))
  done
  sleep 60
  while true; do
    ct=0; tot=$i
    j=0
    for f in $(ssh -o ConnectTimeout=10 $BOX "ls /tmp/mini_tail_ck_*"); do
      s=$(echo $SPARKS | cut -d" " -f$(( (j % 16) + 1 )))
      ssh -o ConnectTimeout=8 -o BatchMode=yes $s "grep -q COMPLETE ~/$f.log 2>/dev/null" && ct=$((ct+1))
      j=$((j+1))
    done
    log "tail chunks: $ct/$tot"
    [ "$ct" -eq "$tot" ] && break
    sleep 120
  done
  for f in $(ssh -o ConnectTimeout=10 $BOX "ls /tmp/mini_tail_ck_*"); do
    s=$(echo $SPARKS | cut -d" " -f$(( (RANDOM % 16) + 1 )))
    ssh -o ConnectTimeout=8 -o BatchMode=yes $s "cat ~/$f.out" >> /extnvme/active/miniature_labels.tsv 2>/dev/null
  done
fi
log "final labels: $(wc -l < /extnvme/active/miniature_labels.tsv)"

# 4) finish: pack miniatures into bins, then provenance
ssh -o ConnectTimeout=10 $BOX "nice -n 10 python3 /tmp/assemble16.py && nice -n 10 python3 /tmp/build_provenance.py && nice -n 10 python3 /tmp/load_provenance.py"
log "BUILD COMPLETE"
