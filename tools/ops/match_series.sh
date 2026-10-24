#!/bin/bash
# ONE match runner for every series. The series name, the subject net, and
# the opponents are each defined exactly once; all engine names, file names,
# and logs derive from them. No per-series copies, no string surgery.
#
# Usage: match_series.sh <SERIES> <NET_PATH> [opponent ...]
#   opponent keys (default: general3 general3b SF8 SF17):
#     general3 general3b SF8 SF17 nQ   (engines resolved from CHESS_DATA)
# Env: CHESS_DATA FASTCHESS_BIN FORK_BIN SF17_BIN   (all required)
# Protocols: 2s (timemargin 200) + d20 for every opponent; SF8 is d20-only
# on ARM (predates ARM SIMD — a fair time control is impossible there).
set -u
: "${CHESS_DATA:?}" "${FASTCHESS_BIN:?}" "${FORK_BIN:?}" "${SF17_BIN:?}"
SERIES="$1"; NET="$2"; shift 2
OPPS="${*:-general3 general3b SF8 SF17}"
[ -f "$NET" ] || { echo "net missing: $NET"; exit 1; }
OUT="$CHESS_DATA/matches"
mkdir -p "$OUT"
NQF="option.EvalFile=$CHESS_DATA/engines/nq13/nets/balanced_l0.nnue option.EvalFile2=$CHESS_DATA/engines/nq13/nets/balanced_l1.nnue option.EvalFile3=$CHESS_DATA/engines/nq13/nets/balanced_l2.nnue option.EvalFile4=$CHESS_DATA/engines/nq13/nets/balanced_l3.nnue option.EvalFile5=$CHESS_DATA/engines/nq13/nets/nvb.nnue option.EvalFile6=$CHESS_DATA/engines/nq13/nets/nvr.nnue option.EvalFile7=$CHESS_DATA/engines/nq13/nets/bvr.nnue option.EvalFile8=$CHESS_DATA/engines/nq13/nets/rv2m.nnue option.EvalFile9=$CHESS_DATA/engines/nq13/nets/qvmat.nnue option.EvalFile10=$CHESS_DATA/engines/nq13/nets/oppb.nnue option.EvalFile11=$CHESS_DATA/engines/nq13/nets/dvoretsky.nnue option.EvalFile12=$CHESS_DATA/engines/nq13/nets/exchanges.nnue option.EvalFile13=$CHESS_DATA/engines/nq13/nets/tactics.nnue"

# opponent key -> engine command | uci options | protocols
opp() {
  case "$1" in
    general3)  echo "$FORK_BIN|option.EvalFile=$CHESS_DATA/runs-general3/best.nnue|2s d20" ;;
    general3b) echo "$FORK_BIN|option.EvalFile=$CHESS_DATA/runs-general3b/best.nnue|2s d20" ;;
    SF8)       echo "$CHESS_DATA/engines/sf8/stockfish||d20" ;;
    SF17)      echo "$SF17_BIN||2s d20" ;;
    nQ)        echo "$CHESS_DATA/engines/nq13/stockfish|$NQF|2s d20" ;;
    *)         echo "unknown opponent: $1" >&2; return 1 ;;
  esac
}

for OPP in $OPPS; do
  IFS='|' read -r OCMD OOPT PROTOS ODDS <<< "$(opp "$OPP")" || exit 1
  for PROTO in $PROTOS; do
    PGN="$OUT/${SERIES}_vs_${OPP}_${PROTO}.pgn"
    # idempotent per series — but only COMPLETE matches (6 games) count;
    # empty/partial leftovers from crashed runs are redone
    if [ -f "$PGN" ] && [ "$(grep -c '\[Event' "$PGN" 2>/dev/null || echo 0)" -ge 6 ]; then
      continue
    fi
    if [ "$PROTO" = 2s ]; then EACH="proto=uci timemargin=200"
    else TC="depth=20"; OTCS=""; EACH="proto=uci"; fi
    rm -f "$PGN"
    if [ "$PROTO" = 2s ]; then TC="st=2"; OTCS="st=${ODDS}"; fi
    echo "[$(date +%H:%M)] $SERIES vs $OPP $PROTO (opp $OTCS) — match starting"
    "$FASTCHESS_BIN" \
      -engine name="$SERIES" cmd="$FORK_BIN" $TC option.EvalFile="$NET" \
      -engine name="$OPP" cmd="$OCMD" $OTCS $OOPT \
      -each $EACH -rounds 3 -repeat -recover \
      -openings file="$CHESS_DATA/engines/3open.epd" format=epd \
      -pgnout file="$PGN" \
      2>&1 | tee "$OUT/${SERIES}_vs_${OPP}_${PROTO}.log" | grep -E "Points" | tail -1
    echo "=== $SERIES vs $OPP $PROTO: $(grep Points "$OUT/${SERIES}_vs_${OPP}_${PROTO}.log" | tail -1) ==="
  done
done
echo "ALL-${SERIES}-MATCHES-DONE"
