#!/bin/bash
# Serialize the 5 experts whose 30-min window ended before epoch 1 completed
# (tiny checkpoints, interpreter shutdown race). Uses last.ckpt or the
# newest intact epoch ckpt, then re-runs the match.
cd /srv/workspace/flychess/src/nnue-pytorch
R=/srv/workspace/chess-active/runs-correct16
for e in nvb dv_rest op_even_l0 op_even_l1 op_even_l2p; do
  [ -s $R/$e/best.nnue ] && { echo "$e ok"; continue; }
  CK=$(python3 - <<EOF
import glob, re, zipfile
cks = glob.glob("$R/$e/lightning_logs/version_*/checkpoints/*.ckpt")
def ep(f):
    m = re.search(r"epoch=(\d+)", f)
    s = re.search(r"step=(\d+)", f)
    if m and s:
        return (int(m.group(1)), int(s.group(1)))
    return (-1, -1)
# prefer last.ckpt if intact, else newest epoch ckpt
for c in [x for x in cks if "last.ckpt" in x] + sorted(
        [x for x in cks if "epoch=" in x], key=ep, reverse=True):
    try:
        zipfile.ZipFile(c).namelist()
        print(c); break
    except Exception:
        continue
EOF
)
  if [ -n "$CK" ]; then
    python3 serialize.py "$CK" $R/$e/best.nnue >/dev/null 2>&1 \
      && echo "$e serialized from ${CK##*/}" \
      || echo "$e SERIALIZE FAILED"
  else
    echo "$e NO INTACT CKPT"
  fi
done
ls -la $R/*/best.nnue | awk '$5 == 0 {print "STILL EMPTY:", $9}'
echo "=== reassembling engine + rerunning matches ==="
E=/srv/workspace/chess-active/engine16-correct
for e in tb mvr rv2m qvmat nvb piece_down oppb dv_Q dv_R dv_rest op_pawnimb op_even_l0 op_even_l1 op_even_l2p mg_unsafe mg_safe; do
  cp $R/$e/best.nnue $E/$e.nnue
done
ls -la $E/*.nnue | awk '$5 == 0 {print "ENGINE EMPTY:", $9}'
bash /srv/workspace/chess-active/match_correct16_run.sh
