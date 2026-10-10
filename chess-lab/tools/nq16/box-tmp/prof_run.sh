#!/bin/bash
ACT=/srv/workspace/flychess/src/Stockfish-act/src/stockfish
export PHASE_MOE=$1
perf record -F 400 -o /tmp/perf_$1.data -- $ACT < <( cat /tmp/prof_cmd.txt; printf "isready\nposition startpos moves e2e4 e7e5\ngo movetime 5000\nquit\n" ) > /dev/null 2>&1
