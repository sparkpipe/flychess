#!/bin/bash
# Retrain the 4 experts whose ckpts were corrupted by the disk-full window.
# 20 min each (these bins are 0.4-0.7M rows), then rebuild the engine and
# run all four matches.
cd /srv/workspace/flychess/src/nnue-pytorch
B=/srv/workspace/chess-active/bins-correct16
R=/srv/workspace/chess-active/runs-correct16
for e in dv_rest op_even_l0 op_even_l1 op_even_l2p; do
  rm -rf $R/$e
  mkdir -p $R/$e
  TR=$(( $(stat -c %s $B/$e.train.bin) / 40 ))
  VR=$(( $(stat -c %s $B/$e.val.bin) / 40 )); [ $VR -gt 30000 ] && VR=30000
  echo "[$(date +%H:%M)] $e retrain (epoch-size $TR)"
  python3 train.py $B/$e.train.bin --validation-datasets $B/$e.val.bin \
    --validation-size $VR --check-val-every-n-epoch 1 --epoch-size $TR \
    --batch-size 4096 --max-time 00:00:20:00 --max-epochs 100000 \
    --network-save-period 1 --save_top_k 1 \
    --random-fen-skipping 3 --default-root-dir $R/$e >> $R/$e/train.log 2>&1
  CK=$(python3 - <<PYEOF
import glob, re, zipfile
cks = glob.glob("$R/$e/lightning_logs/version_*/checkpoints/epoch=*.ckpt")
def ep(f):
    m = re.search(r"epoch=(\d+)", f)
    return int(m.group(1)) if m else -1
cks.sort(key=ep, reverse=True)
for c in cks:
    try:
        zipfile.ZipFile(c).namelist()
        print(c); break
    except Exception:
        continue
PYEOF
)
  [ -n "$CK" ] && python3 serialize.py "$CK" $R/$e/best.nnue >/dev/null 2>&1 && echo "$e OK" || echo "$e FAILED"
done
E=/srv/workspace/chess-active/engine16-correct
for e in tb mvr rv2m qvmat nvb piece_down oppb dv_Q dv_R dv_rest op_pawnimb op_even_l0 op_even_l1 op_even_l2p mg_unsafe mg_safe; do
  cp $R/$e/best.nnue $E/$e.nnue
done
ls -la $E/*.nnue | awk '$5==0 {print "EMPTY:", $9}'
echo "engine assembled — running matches"
bash /srv/workspace/chess-active/match_correct16_run.sh
