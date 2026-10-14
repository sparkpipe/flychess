#!/bin/bash
# Build general2: concatenate all 16 corrected expert bins into one training
# set — a single general net on clean labels (no routing, no MoE).
set -e
B=/srv/workspace/chess-active/bins-correct16
O=/srv/workspace/chess-active/bins-general2
mkdir -p $O
> $O/general2.train.bin
> $O/general2.val.bin
TR=0; VA=0
for e in tb mvr rv2m qvmat nvb piece_down oppb dv_Q dv_R dv_rest op_pawnimb op_even_l0 op_even_l1 op_even_l2p mg_unsafe mg_safe; do
  cat $B/$e.train.bin >> $O/general2.train.bin
  cat $B/$e.val.bin >> $O/general2.val.bin
  T=$(( $(stat -c %s $B/$e.train.bin) / 40 ))
  V=$(( $(stat -c %s $B/$e.val.bin) / 40 ))
  TR=$((TR+T)); VA=$((VA+V))
  echo "$e: +$T train +$V val"
done
echo "TOTAL: $TR train, $VA val"
echo "sha256:"; sha256sum $O/general2.train.bin $O/general2.val.bin
