#!/bin/bash
# Reclaim chess-era disk on one spark. Run AFTER backup_nets_final verified.
# Keeps: metrics.csv, train.log, watch.log (KBs, autopsy value). Deletes: ckpts, nets
# (backed to cold RAID + picks in engine16/engine16anti), bin copies (regenerable), d20
# files (collected), helper scripts (in repo).
before=$(df /home | tail -1 | awk '{print $4}')
rm -rf ~/run16_*/lightning_logs/version_*/checkpoints ~/runanti_*/lightning_logs/version_*/checkpoints
rm -rf ~/run16_*/nets ~/runanti_*/nets
rm -f  ~/run16_*/checkpoint.*.ckpt ~/runanti_*/checkpoint.*.ckpt
rm -f  ~/anti_*.train.bin ~/anti_*.val.bin
rm -f  ~/tb.train.bin ~/tb.val.bin ~/mvr.*.bin ~/rv2m.*.bin ~/qvmat.*.bin ~/nvb.*.bin \
       ~/piece_down.*.bin ~/oppb.*.bin ~/dv_Q.*.bin ~/dv_R.*.bin ~/dv_rest.*.bin \
       ~/op_pawnimb.*.bin ~/op_even_l0.*.bin ~/op_even_l1.*.bin ~/op_even_l2p.*.bin \
       ~/mg_unsafe.*.bin ~/mg_safe.*.bin
rm -f  ~/d20out.tsv ~/d20job.txt
rm -f  ~/cur16.py ~/pick16.py ~/curanti.py ~/pickanti.py ~/free_sparks.sh
rm -f  /tmp/watch_*.sh /tmp/anti_watch_spark.sh /tmp/launch_anti.sh /tmp/net_prune.sh \
       /tmp/kill16main.sh /tmp/free_sparks.sh /tmp/disk_audit16.sh /tmp/train_watch_spark.sh
after=$(df /home | tail -1 | awk '{print $4}')
echo "$(hostname): freed $(( (after - before) / 1024 / 1024 )) GB; left: $(du -sh ~/run16_* ~/runanti_* 2>/dev/null | awk '{s=$1} END {print s}') of run dirs"
ls ~/run16_* 2>/dev/null | head -3
echo "chess procs now: $(pgrep -f 'train\.py|train23|watch|d20_spark' | wc -l)"
