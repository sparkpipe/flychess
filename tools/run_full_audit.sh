#!/bin/bash
# Run the 100% bin audit as soon as ASSEMBLE_COMPLETE appears.
R=/mnt/cold-raid6/chess-audit
while [ ! -f "$R/ASSEMBLE_COMPLETE" ]; do sleep 120; done
cd /home/spec/chess-lab
python3 tools/audit_all_bins.py >> "$R/audit_run.log" 2>&1 \
  && touch "$R/AUDIT_COMPLETE" || touch "$R/AUDIT_FAILED"
