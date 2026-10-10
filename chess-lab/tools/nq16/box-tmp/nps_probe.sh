#!/bin/bash
E=/mnt/cold-raid6/chess-audit/engine23
N=/mnt/cold-raid6/chess-audit/nets
ACT=/srv/workspace/flychess/src/Stockfish-act/src/stockfish
FORK=/srv/workspace/flychess/src/Stockfish/src/stockfish
probe() {
  local NAME=$1 CMD=$2; shift 2
  printf "setoption name Threads value 1\nposition startpos moves e2e4 e7e5\ngo movetime 3000\nquit\n" | timeout 30 $CMD "$@" 2>/dev/null | grep -a "info depth" | tail -1 | grep -ao "nodes [0-9]* nps [0-9]*" | sed "s/^/$NAME: /"
}
probe nQ23 $ACT
probe nQ $FORK
probe SF8 /usr/games/stockfish
