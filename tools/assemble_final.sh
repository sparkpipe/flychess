#!/bin/bash
# FINAL ASSEMBLY: after BOTH_COMPLETE ->
#   1. repack tactics.bin (was parked in the purged dir; evals intact)
#   2. run the gambit-augmentation evals alone on the box (8 workers)
#   3. classify+pack aug positions per expert (aug_<expert>.bin additions)
set -u
R=/mnt/cold-raid6/chess-audit
LOG="$R/assemble.log"
exec >> "$LOG" 2>&1
echo "=== assemble armed $(date -u) ==="

while [ ! -f "$R/BOTH_COMPLETE" ]; do sleep 120; done
echo "BOTH_COMPLETE seen $(date -u)"

cd /home/spec/chess-lab
python3 tools/pack_tactics_bin.py "$R/tactics_evals" \
  "$R/expert_bins_both/tactics.bin" > "$R/tactics_pack2.txt" 2>&1 \
  || { touch "$R/ASSEMBLE_FAILED"; exit 1; }
tail -1 "$R/tactics_pack2.txt"

rm -f "$R"/gaug_evals/w*.txt "$R"/gaug_evals/w*.log
for W in 0 1 2 3 4 5 6 7; do
  WORKER_ID=$W NUM_WORKERS=8 DEPTH=12 \
    INPUT=$R/gambit_aug_positions.txt \
    OUTPUT=$R/gaug_evals/w$W.txt \
    setsid nohup nice -n 8 python3 tools/eval_local.py \
    > "$R/gaug_evals/w$W.log" 2>&1 < /dev/null &
done
while true; do
  d=$(grep -lh DONE "$R"/gaug_evals/w*.log 2>/dev/null | wc -l)
  [ "$d" -ge 8 ] && break
  sleep 120
done
echo "aug evals done $(date -u)"

cat "$R"/gaug_evals/w*.txt > "$R/gaug_evals_all.txt"
python3 - "$R" << 'PYEOF' > "$R/aug_pack.txt" 2>&1 || exit 1
import sys, glob, struct
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
from pack_expert_bins import assign, pack_sfen, pack_move

R = sys.argv[1]
out = {}
counts = {}
n = 0
for path in sorted(glob.glob(R + "/gaug_evals/w*.txt")):
    for line in open(path):
        p = line.rstrip("\n").split("|")
        if len(p) < 8 or p[7] in ("", "None"):
            continue
        # minimal record with the fields assign() needs
        rec = {"config_i": None, "men": 0, "exch": 0, "ply": int(p[3]),
               "lock_c": 0, "bishops_i": "", "cp": int(p[7]), "fen": p[0]}
        try:
            b = chess.Board(p[0])
            mv = chess.Move.from_uci(p[1])
            if mv not in b.legal_moves:
                continue
            # build config/bishops/contact/men from the board
            w, bl = [], []
            wbc, bbc = [], []
            for s, pc in b.piece_map().items():
                u = pc.symbol().upper()
                if u == "K":
                    continue
                (w if pc.color else bl).append(u)
                if u == "B":
                    (wbc if pc.color else bbc).append(
                        (chess.square_file(s) + chess.square_rank(s)) % 2)
            order = {"Q": 0, "R": 1, "B": 2, "N": 3, "P": 4}
            cw = "".join(sorted(w, key=lambda x: order[x]))
            cb = "".join(sorted(bl, key=lambda x: order[x]))
            rec["config_i"] = cw + "v" + cb
            rec["men"] = len(b.piece_map())
            if wbc and bbc and \
                    (sum(wbc) * 2 > len(wbc)) != (sum(bbc) * 2 > len(bbc)):
                rec["bishops_i"] = "opp_bishops"
            lock = 0
            for f in range(8):
                wr = [chess.square_rank(s) for s in chess.SquareSet(
                    b.pieces(chess.PAWN, chess.WHITE))
                    if chess.square_file(s) == f]
                br = [chess.square_rank(s) for s in chess.SquareSet(
                    b.pieces(chess.PAWN, chess.BLACK))
                    if chess.square_file(s) == f]
                if wr and br and min(br) - max(wr) == 1 and 2 <= f <= 5:
                    lock += 1
            rec["lock_c"] = lock
            ex = assign(rec)
            if ex == "tb-region":
                continue
            blob = struct.pack("<32shHHbB", pack_sfen(b), rec["cp"],
                               pack_move(mv), b.fullmove_number, 0, 0)
            if ex not in out:
                out[ex] = open("%s/expert_bins_both/aug_%s.bin" % (R, ex), "wb")
            out[ex].write(blob)
            counts[ex] = counts.get(ex, 0) + 1
            n += 1
        except Exception:
            continue
for fh in out.values():
    fh.close()
print("AUG PACKED: %d positions" % n)
for e, c in sorted(counts.items(), key=lambda kv: -kv[1]):
    print("  %-14s %10s" % (e, format(c, ",")))
print("ASSEMBLE-AUG-COMPLETE")
PYEOF
touch "$R/ASSEMBLE_COMPLETE"
echo "=== ASSEMBLE COMPLETE $(date -u) ==="
