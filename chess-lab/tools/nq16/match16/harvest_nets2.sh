#!/bin/bash
# Pull best anti nets sequentially (parallel variant collided).
AN=/srv/workspace/chess-active/engine16anti
MAP="spark0:tb spark1:mvr spark2:rv2m spark3:qvmat spark4:nvb spark5:piece_down spark6:oppb \
spark7:dv_Q spark8:dv_R spark9:dv_rest sparka:op_pawnimb sparkb:op_even_l0 \
sparkd:op_even_l2p sparke:mg_unsafe sparkf:mg_safe"
for pair in $MAP; do
  s=${pair%%:*}
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $s "python3 ~/pickanti.py" 2>/dev/null)
  for e_ep in $out; do :; done
  echo "$out" | while read e ep corr; do
    [ -z "$e" ] && continue
    rm -f $AN/$e.nnue
    if scp -q -o BatchMode=yes -o ConnectTimeout=20 $s:runanti_$e/nets/${e}_e$ep.nnue $AN/$e.nnue; then
      sz=$(stat -c %s $AN/$e.nnue)
      [ "$sz" -gt 80000000 ] && echo "$s: $e e$ep corr=$corr OK ($sz bytes)" || echo "$s: $e e$ep SIZE BAD ($sz)"
    else
      echo "$s: $e e$ep PULL FAIL"
    fi
  done
done
echo done
