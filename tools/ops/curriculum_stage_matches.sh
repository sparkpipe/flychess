#!/bin/bash
# Curriculum per-stage engines + nQ progression matches.
# For each completed curriculum stage dD (next stage's dir exists, or d20
# finished): serialize the stage's best checkpoint to stage_dD/best.nnue,
# then play 6-game matches vs nQ at 2s/move (timemargin 200) and fixed
# depth 20. Idempotent: a stage with both PGNs is skipped.
: "${CHESS_DATA:?CHESS_DATA not set (data root)}"
: "${FASTCHESS_BIN:?FASTCHESS_BIN not set (fastchess binary)}"
: "${FORK_BIN:?FORK_BIN not set (n16-fork stockfish)}"
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
R="$CHESS_DATA/runs-general3c"
ENG="$CHESS_DATA/engines"
NQ="$ENG/nq13/stockfish"
OPEN="$ENG/3open.epd"
OUT="$CHESS_DATA/matches"
mkdir -p "$OUT"
[ -x "$NQ" ] || { echo "nQ binary missing: $NQ"; exit 1; }
[ -f "$OPEN" ] || { echo "openings missing: $OPEN"; exit 1; }

NN="$ENG/nq13/nets"
NQF="option.EvalFile=$NN/balanced_l0.nnue option.EvalFile2=$NN/balanced_l1.nnue option.EvalFile3=$NN/balanced_l2.nnue option.EvalFile4=$NN/balanced_l3.nnue option.EvalFile5=$NN/nvb.nnue option.EvalFile6=$NN/nvr.nnue option.EvalFile7=$NN/bvr.nnue option.EvalFile8=$NN/rv2m.nnue option.EvalFile9=$NN/qvmat.nnue option.EvalFile10=$NN/oppb.nnue option.EvalFile11=$NN/dvoretsky.nnue option.EvalFile12=$NN/exchanges.nnue option.EvalFile13=$NN/tactics.nnue"

best_ckpt() {  # newest zip-intact epoch=*.ckpt of a stage
  python3 - "$1" << 'EOF'
import glob, re, sys, zipfile
cks = glob.glob(f"{sys.argv[1]}/lightning_logs/version_*/checkpoints/epoch=*.ckpt")
def ep(f):
    m = re.search(r"epoch=(\d+)", f); return int(m.group(1)) if m else -1
cks.sort(key=ep, reverse=True)
for c in cks:
    try:
        zipfile.ZipFile(c).namelist(); print(c); break
    except Exception:
        continue
EOF
}

while true; do
  ALLDONE=1
  for D in 10 11 12 13 14 15 16 17 18 19 20; do
    [ -f "$OUT/d${D}_vs_nq_d20.pgn" ] && continue
    ALLDONE=0
    SR="$R/stage_d$D"
    if [ "$D" -lt 20 ]; then
      [ -d "$R/stage_d$((D+1))" ] || continue      # stage D complete iff next started
    else
      [ -f "$R/best.nnue" ] || continue             # d20 complete iff chain done
    fi
    CK=$(best_ckpt "$SR")
    [ -z "$CK" ] && { echo "d$D: no intact ckpt"; continue; }
    if [ ! -f "$SR/best.nnue" ] || [ "$SR/best.nnue" -ot "$CK" ]; then
      echo "[$(date +%H:%M)] d$D: serializing $(basename $CK)"
      (cd "$REPO/nnue-pytorch" && python3 serialize.py "$CK" "$SR/best.nnue") \
        || { echo "d$D serialize FAILED"; continue; }
    fi
    NET="option.EvalFile=$SR/best.nnue"
    for PROTO in 2s d20; do
      [ -f "$OUT/d${D}_vs_nq_${PROTO}.pgn" ] && continue
      if [ "$PROTO" = 2s ]; then TC="st=2"; EACH="proto=uci timemargin=200"; else TC="depth=20"; EACH="proto=uci"; fi
      rm -f "$OUT/d${D}_vs_nq_${PROTO}.pgn"
      echo "[$(date +%H:%M)] d$D vs nQ $PROTO — match starting"
      "$FASTCHESS_BIN" \
        -engine name=d$D cmd="$FORK_BIN" $TC $NET \
        -engine name=nQ cmd="$NQ" $TC $NQF \
        -each $EACH -rounds 3 -repeat -recover \
        -openings file="$OPEN" format=epd \
        -pgnout file="$OUT/d${D}_vs_nq_${PROTO}.pgn" \
        2>&1 | tee "$OUT/d${D}_vs_nq_${PROTO}.log" | grep -E "Points|Score of" | tail -2
      echo "=== d$D vs nQ $PROTO: $(grep Points "$OUT/d${D}_vs_nq_${PROTO}.log" | tail -1) ==="
    done
  done
  [ "$ALLDONE" = 1 ] && { echo "[$(date +%H:%M)] all curriculum nQ matches DONE"; exit 0; }
  sleep 300
done
