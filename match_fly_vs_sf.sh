#!/bin/bash
# FlyLeela (fly-brain MCTS, 5s/move) vs Stockfish (~2-ply, 50ms/move)
# Openings: Noomen's Daring gambit book — the fly plays gambits.
cd /home/spec/chess-lab
FC=/home/spec/chess-lab/fastchess-linux-x86-64/fastchess
ROUNDS=${1:-6}
$FC \
  -engine cmd=python3 args=/home/spec/chess-lab/fly_mcts.py name=FlyLeela tc=5+0.2 \
  -engine cmd=/usr/games/stockfish name=SF2 tc=0.05+0.01 option.Threads=1 option.Hash=64 \
  -each proto=uci \
  -openings file=/home/spec/chess-lab/Daring.bin format=bin policy=weighted order=random \
  -rounds "$ROUNDS" -games 2 -repeat 0 \
  -pgnout file=fly_vs_sf2.pgn format=pgn 2>&1 | tail -30
