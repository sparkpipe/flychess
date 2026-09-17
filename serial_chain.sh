#!/bin/bash
cd /home/spec/chess-lab
echo "=== confirm 6 core batteries on restored base ===" > serial.log
python3 - << "PYEOF" >> serial.log 2>&1
import sys, torch, chess
sys.path.insert(0, "/home/spec/chess-lab")
import fly_curriculum as cur, flyfeat_cb
torch.manual_seed(0)
flyfeat_cb.feat_vec(chess.Board())
v0, NAMES = flyfeat_cb.feat_vec(chess.Board())
m = cur.FlyCB(len(flyfeat_cb.FEATURE_KEYS)).to("cuda")
m.load_state_dict(torch.load("/home/spec/chess-lab/fly_cb_m1.pt", weights_only=True))
m.eval()
for p in cur.PIECE_ORDER:
    pr, _ = cur.eval_piece(m, p, n=64, stage=1)
    print(chess.piece_name(p), round(pr, 3), flush=True)
PYEOF
for spec in "ep PIECES=pawn,ep" "promo PIECES=pawn,promo" "castle PIECES=king,rook,castle"; do
  set -- $spec
  echo "=== SERIAL MILESTONE $1 ===" >> serial.log
  STATE=/home/spec/chess-lab/fly_cb_m1.pt RETINO=geo READOUT=variance \
    env $2 nohup python3 fly_curriculum.py 1 --steps 4000 > serial_$1.out 2>&1
  grep -q "STAGE 1 ALL MILESTONES PASSED" serial_$1.out || { echo "SERIAL-FAIL at $1" >> serial.log; exit 1; }
done
echo "=== SERIAL 9-ARM COMPLETE ===" >> serial.log
