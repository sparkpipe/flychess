#!/bin/bash
# Disk reclaim: keep every best.nnue + final ckpts, delete bulky intermediates.
R=/srv/workspace/chess-active
before=$(df /srv/workspace | tail -1 | awk '{print $4}')
# 1. pilot_run: nets already extracted to engine16-quick era; whole dir is pilot debris
rm -rf $R/pilot_run
# 2. old label tsvs (superseded by DB; wrong-corpus artifacts)
rm -f $R/miniature_rows.tsv $R/miniature_labels.tsv $R/miniature_labels17.tsv.part
# 3. runs-10min/extra10: keep final.nnue, drop ckpts/logs (nets also re-derivable from DB)
for D in runs-10min runs-extra10 runs-continue; do
  find $R/$D -name "*.ckpt" -delete 2>/dev/null
done
# 4. runs-fleet1h: keep best.nnue + ONE newest intact ckpt per expert
for e in $(ls $R/runs-fleet1h | grep -v progress); do
  CKS=$(ls $R/runs-fleet1h/$e/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | sort -t= -k2 -n)
  keep=$(echo "$CKS" | tail -1)
  for c in $CKS; do [ "$c" != "$keep" ] && rm -f "$c"; done
done
# 5. old export dirs superseded by bins-fleet-v1 manifest copies on cold raid
rm -rf $R/bins-db-v2
after=$(df /srv/workspace | tail -1 | awk '{print $4}')
echo "freed $(( (after - before) / 1024 / 1024 )) GB -> $(df -h /srv/workspace | tail -1 | awk '{print $4}') free"
