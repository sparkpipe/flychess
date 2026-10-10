#!/bin/bash
# Harvest anti16 best nets + d20 outputs from sparks (runs ON box, before freeing sparks).
DD=/mnt/cold-raid6/chess-audit/depth_db
AN=/srv/workspace/chess-active/engine16anti
mkdir -p $DD/shards $AN
MAP="spark0:tb spark1:mvr spark2:rv2m spark3:qvmat spark4:nvb spark5:piece_down spark6:oppb \
spark7:dv_Q spark8:dv_R spark9:dv_rest sparka:op_pawnimb sparkb:op_even_l0 \
sparkd:op_even_l2p sparke:mg_unsafe sparkf:mg_safe"
for pair in $MAP; do
  s=${pair%%:*}
  (
    scp -q -o BatchMode=yes -o ConnectTimeout=15 $s:d20out.tsv $DD/shards/d20_$s.tsv 2>/dev/null \
      && echo "$s: d20 pulled ($(wc -l < $DD/shards/d20_$s.tsv) lines)" || echo "$s: D20 FAIL"
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $s "python3 ~/pickanti.py" 2>/dev/null)
    echo "$out" | while read e ep corr; do
      [ -z "$e" ] && continue
      if scp -q -o BatchMode=yes -o ConnectTimeout=15 $s:runanti_$e/nets/${e}_e$ep.nnue $AN/$e.nnue 2>/dev/null; then
        echo "$s: $e best e$ep corr=$corr pulled"
      else
        echo "$s: $e e$ep NET PULL FAIL"
      fi
    done
  ) &
done
wait
echo "== harvest summary =="
wc -l $DD/shards/d20_*.tsv | tail -3
ls -la $AN/
