#!/bin/bash
# 6-lesson curriculum restart (operator spec 2026-09-16):
#   L1 rook | L2 bishop | L3 queen | L4 knight | L5 pawn+ep+promo | L6 king+castle
# Serial. Each lesson resumes from the PREVIOUS lesson's checkpoint (copied
# into its own STATE name first — fly_curriculum.py loads AND saves STATE),
# so cumulative piece lists re-verify at step 1 and only the new piece
# trains. Each PASS gate (pair>=0.99) fires the regression sweep of all
# prior batteries with corrective boosts; stalls do stop-dump-boost-
# continue inside the run. Idempotent: a lesson with a PASSED out file and
# its checkpoint on disk is skipped on relaunch.
cd /home/spec/chess-lab
LOG=lessons6.log
echo "=== LESSONS6 START $(date -Is) ===" >> "$LOG"

lesson () {  # name pieces state [fresh] [prev_ckpt]
  local name="$1" pieces="$2" state="$3" fresh="${4:-}" prev="${5:-}"
  local ckpt="/home/spec/chess-lab/$state"
  if grep -q "STAGE 1 ALL MILESTONES PASSED" "lesson_$name.out" 2>/dev/null \
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
      python3 fly_curriculum.py 1 --steps 4000 > "lesson_$name.out" 2>&1
  else
    STATE="$ckpt" PIECES="$pieces" RETINO=geo READOUT=variance \
      python3 fly_curriculum.py 1 --steps 4000 > "lesson_$name.out" 2>&1
  fi
  if grep -q "STAGE 1 ALL MILESTONES PASSED" "lesson_$name.out"; then
    echo "LESSON $name PASS" >> "$LOG"
    grep "REGRESSION-SWEEP" "lesson_$name.out" | tail -1 >> "$LOG"
  else
    echo "LESSON $name FAIL — see lesson_$name.out" >> "$LOG"
    exit 1
  fi
}

lesson l1_rook   "rook" fly_cb_v2_l1_rook.pt fresh
lesson l2_bishop "rook,bishop" fly_cb_v2_l2_bishop.pt "" fly_cb_v2_l1_rook.pt
lesson l3_queen  "rook,bishop,queen" fly_cb_v2_l3_queen.pt "" fly_cb_v2_l2_bishop.pt
lesson l4_knight "rook,bishop,queen,knight" fly_cb_v2_l4_knight.pt "" fly_cb_v2_l3_queen.pt
lesson l5_pawn   "rook,bishop,queen,knight,pawn,ep,promo" fly_cb_v2_l5_pawn.pt "" fly_cb_v2_l4_knight.pt
lesson l6_king   "rook,bishop,queen,knight,pawn,ep,promo,king,castle" fly_cb_v2_l6_king.pt "" fly_cb_v2_l5_pawn.pt

# Final regression test from checkpoint using corrections: reload the final
# checkpoint in a fresh process; all 9 batteries must re-pass the 0.99 gate
# (milestone machinery trains + applies slot corrections on any that dip).
echo "=== FINAL REGRESSION FROM CHECKPOINT (9 batteries) ===" >> "$LOG"
STATE="/home/spec/chess-lab/fly_cb_v2_l6_king.pt" \
  PIECES="rook,bishop,queen,knight,pawn,ep,promo,king,castle" \
  RETINO=geo READOUT=variance \
  python3 fly_curriculum.py 1 --steps 4000 > lessons6_confirm.out 2>&1
if grep -q "STAGE 1 ALL MILESTONES PASSED" lessons6_confirm.out; then
  echo "=== LESSONS6 COMPLETE — ALL 9 BATTERIES PASS FROM CHECKPOINT ===" >> "$LOG"
else
  echo "=== LESSONS6 CONFIRM FAIL — see lessons6_confirm.out ===" >> "$LOG"
  exit 1
fi
