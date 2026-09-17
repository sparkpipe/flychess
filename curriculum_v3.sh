#!/bin/bash
# FULL CURRICULUM v3 (operator directive 2026-09-16: "retrain from the
# beginning" — with the piece-conditioned move view baked in from step 0).
# Order: L1 rook | L2 bishop | L3 queen | L4 knight | L5 pawn+ep+promo |
#        L6 king+castle   (stage 1 lessons, cumulative PIECES lists)
#      -> S2 blocking/capture -> S3 protection -> S4 king safety/pin/fork/
#         discovered.
# Serial-resume: each step copies the previous checkpoint into its own STATE
# (fly_curriculum loads AND saves STATE); idempotent skip when the out file
# shows PASSED and the checkpoint exists. Gates 0.98; stop-correct-continue
# in-run; mix-replay of taught batteries; 110-step maintenance with
# fail-fast; stable (_phash) seeds everywhere.
cd /home/spec/chess-lab
LOG=curriculum_v3.log
echo "=== CURRICULUM V3 START $(date -Is) ===" >> "$LOG"

run1 () {  # name pieces state fresh prev
  local name="$1" pieces="$2" state="$3" fresh="${4:-}" prev="${5:-}"
  local ckpt="/home/spec/chess-lab/$state"
  if grep -q "STAGE 1 ALL MILESTONES PASSED" "v3_$name.out" 2>/dev/null \
     && [ -f "$ckpt" ]; then
    echo "LESSON $name already PASSED (skip)" >> "$LOG"
    return 0
  fi
  if [ -n "$prev" ] && [ -f "/home/spec/chess-lab/$prev" ] && [ ! -f "$ckpt" ]; then
    cp "/home/spec/chess-lab/$prev" "$ckpt"
  fi
  echo "=== LESSON $name PIECES=$pieces -> $state ===" >> "$LOG"
  if [ "$fresh" = "fresh" ]; then
    STATE="$ckpt" PIECES="$pieces" RETINO=geo READOUT=variance FRESH=1 \
      python3 fly_curriculum.py 1 > "v3_$name.out" 2>&1
  else
    STATE="$ckpt" PIECES="$pieces" RETINO=geo READOUT=variance \
      python3 fly_curriculum.py 1 > "v3_$name.out" 2>&1
  fi
  if grep -q "STAGE 1 ALL MILESTONES PASSED" "v3_$name.out"; then
    echo "LESSON $name PASS" >> "$LOG"
    grep "REGRESSION-SWEEP" "v3_$name.out" | tail -1 >> "$LOG"
  else
    echo "LESSON $name FAIL — see v3_$name.out" >> "$LOG"
    exit 1
  fi
}

runN () {  # name stage prev state
  local name="$1" st="$2" prev="$3" state="$4"
  local ckpt="/home/spec/chess-lab/$state"
  if grep -q "STAGE $st ALL MILESTONES PASSED" "v3_$name.out" 2>/dev/null \
     && [ -f "$ckpt" ]; then
    echo "STAGE $st already PASSED (skip)" >> "$LOG"
    return 0
  fi
  if [ -f "/home/spec/chess-lab/$prev" ] && [ ! -f "$ckpt" ]; then
    cp "/home/spec/chess-lab/$prev" "$ckpt"
  fi
  echo "=== STAGE $st -> $state ===" >> "$LOG"
  STATE="$ckpt" PIECES="rook,bishop,queen,knight,pawn,ep,promo,king,castle" \
    RETINO=geo READOUT=variance \
    python3 fly_curriculum.py "$st" > "v3_$name.out" 2>&1
  if grep -q "STAGE $st ALL MILESTONES PASSED" "v3_$name.out"; then
    echo "STAGE $st PASS" >> "$LOG"
    grep "REGRESSION-SWEEP" "v3_$name.out" | tail -1 >> "$LOG"
  else
    echo "STAGE $st FAIL — see v3_$name.out" >> "$LOG"
    exit 1
  fi
}

run1 l1_rook   "rook" fly_cb_v3_l1.pt fresh
run1 l2_bishop "rook,bishop" fly_cb_v3_l2.pt "" fly_cb_v3_l1.pt
run1 l3_queen  "rook,bishop,queen" fly_cb_v3_l3.pt "" fly_cb_v3_l2.pt
run1 l4_knight "rook,bishop,queen,knight" fly_cb_v3_l4.pt "" fly_cb_v3_l3.pt
run1 l5_pawn   "rook,bishop,queen,knight,pawn,ep,promo" fly_cb_v3_l5.pt "" fly_cb_v3_l4.pt
run1 l6_king   "rook,bishop,queen,knight,pawn,ep,promo,king,castle" fly_cb_v3_l6.pt "" fly_cb_v3_l5.pt

runN s2 2 fly_cb_v3_l6.pt fly_cb_v3_s2.pt
runN s3 3 fly_cb_v3_s2.pt fly_cb_v3_s3.pt
runN s4 4 fly_cb_v3_s3.pt fly_cb_v3_s4.pt

echo "=== CURRICULUM V3 COMPLETE — v3 lineage through stage 4 ===" >> "$LOG"
