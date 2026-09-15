#!/bin/bash
# Elo-ladder rung vs the fly. Usage: [SWARM=true|false] ladder_match.sh <rung> [rounds]
# Rungs: 500 | 1000 | 1300 (python anchors) | 1500 | 1800 (Stockfish UCI_Elo)
cd /home/spec/chess-lab
RUNG=${1:-1000}
ROUNDS=${2:-4}
FC=/home/spec/chess-lab/fastchess-linux-x86-64/fastchess

case "$RUNG" in
  500)  OPP="cmd=/home/spec/chess-lab/anchor500.sh name=Anchor500 tc=10+0.2";;
  1000) OPP="cmd=/home/spec/chess-lab/anchor1000.sh name=Anchor1000 tc=10+0.2";;
  1300) OPP="cmd=/home/spec/chess-lab/anchor1300.sh name=Anchor1300 tc=10+0.2";;
  1500) OPP="cmd=/usr/games/stockfish name=SF1500 tc=5+0.1 option.UCI_LimitStrength=true option.UCI_Elo=1500 option.Threads=1";;
  1800) OPP="cmd=/usr/games/stockfish name=SF1800 tc=5+0.1 option.UCI_LimitStrength=true option.UCI_Elo=1800 option.Threads=1";;
  *) echo "rung must be 500|1000|1300|1500|1800"; exit 1;;
esac

$FC \
  -engine cmd=python3 args=/home/spec/chess-lab/fly_mcts.py name=FlyLeela tc=60+0.5 option.Swarm=${SWARM:-true} option.SwarmSeed=${SEED:-7} \
  -engine $OPP \
  -each proto=uci \
  -openings file=/home/spec/chess-lab/gambit_lines.pgn format=pgn order=random \
  -rounds "$ROUNDS" -games 2 \
  -pgnout file=ladder_${RUNG}.pgn notation=san
