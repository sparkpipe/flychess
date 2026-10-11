#!/bin/bash
# Final nets backup to cold RAID BEFORE spark cleanup. Runs ON box.
# Main nets -> train16_backup/<e>/nets/, anti nets -> anti16_backup/<e>/nets/
TB=/mnt/cold-raid6/chess-audit/train16_backup
AB=/mnt/cold-raid6/chess-audit/anti16_backup
MAP="spark0:tb spark1:mvr spark2:rv2m spark3:qvmat spark4:nvb spark5:piece_down spark6:oppb \
spark7:dv_Q spark8:dv_R spark9:dv_rest sparka:op_pawnimb sparkb:op_even_l0 \
sparkd:op_even_l2p sparke:mg_unsafe sparkf:mg_safe"
mkdir -p $TB/op_even_l1/nets $AB/op_even_l1/nets
rsync -a /srv/workspace/chess-active/run16_op_even_l1/nets/ $TB/op_even_l1/nets/ &
rsync -a /srv/workspace/chess-active/runanti_op_even_l1/nets/ $AB/op_even_l1/nets/ &
for pair in $MAP; do
  s=${pair%%:*}; e=${pair##*:}
  mkdir -p $TB/$e/nets $AB/$e/nets
  ( rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=15" $s:run16_$e/nets/ $TB/$e/nets/ \
      && rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=15" $s:runanti_$e/nets/ $AB/$e/nets/ \
      && echo "$s $e: $(ls $TB/$e/nets | wc -l) main + $(ls $AB/$e/nets | wc -l) anti nets backed" ) &
  while [ $(jobs -r | wc -l) -ge 5 ]; do sleep 2; done
done
wait
echo "box op_even_l1: $(ls $TB/op_even_l1/nets | wc -l) main + $(ls $AB/op_even_l1/nets | wc -l) anti"
echo BACKUP-DONE
