#!/bin/bash
ACT=/srv/workspace/flychess/src/Stockfish-act/src/stockfish
export PHASE_MOE=$1
{ cat /tmp/prof_cmd.txt; printf "isready\nposition startpos moves e2e4 e7e5\ngo movetime 5000\nquit\n"; } | $ACT > /dev/null 2>&1
