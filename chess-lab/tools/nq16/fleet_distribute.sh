#!/bin/bash
# Distribute fleet bins to sparks (per-spark expert bins only) + on-spark sha256 audit.
# Runs ON box after db_export_v3 + backups. Harmless to weightd (disk copy only).
set -u
B=/srv/workspace/chess-active/bins-fleet-v1
MAP="spark0:tb spark1:mvr spark2:rv2m spark3:qvmat spark4:nvb spark5:piece_down \
spark6:oppb spark7:dv_Q spark8:dv_R spark9:dv_rest sparka:op_pawnimb \
sparkb:op_even_l0 sparkd:op_even_l2p sparke:mg_unsafe sparkf:mg_safe"
echo "== distributing (box keeps op_even_l1)"
for pair in $MAP; do
  s=${pair%%:*}; e=${pair##*:}
  scp -q -o BatchMode=yes -o ConnectTimeout=15 \
      $B/$e.train.bin $B/$e.val.bin $s: 2>/dev/null \
    && echo "$s <- $e OK" || echo "$s <- $e FAIL"
done
echo "== on-spark sha256 audit (vs manifest)"
for pair in $MAP; do
  s=${pair%%:*}; e=${pair##*:}
  EXP_TR=$(awk -F'\t' -v x=$e '$1==x{print $10}' $B/manifest.tsv)
  EXP_VA=$(awk -F'\t' -v x=$e '$1==x{print $11}' $B/manifest.tsv)
  R=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $s \
      "sha256sum $e.train.bin $e.val.bin 2>/dev/null | awk '{print \$1}'" 2>/dev/null)
  GOT_TR=$(echo "$R" | sed -n 1p); GOT_VA=$(echo "$R" | sed -n 2p)
  if [ "$GOT_TR" = "$EXP_TR" ] && [ "$GOT_VA" = "$EXP_VA" ]; then
    echo "$s: sha256 MATCH ($e)"
  else
    echo "$s: sha256 MISMATCH ($e) got=${GOT_TR:0:12}/${GOT_VA:0:12} want=${EXP_TR:0:12}/${EXP_VA:0:12}"
  fi
done
echo "DISTRIBUTE+AUDIT COMPLETE"
