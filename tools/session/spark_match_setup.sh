#!/bin/bash
# Stage correct16 match assets on spark0 (ARM): engine binaries, nets, book.
# Runs FROM the box; pushes to spark0.
set -e
S=spark0
D=/home/spark0/c16match
ssh -o BatchMode=yes $S "mkdir -p $D/nets $D/nets/nq"
# n16 fork engine (ARM build? check) — the fork is x86-only; but the ARM
# sf17_arm is VANILLA (16-slot fork needed). Check for an ARM n16 build:
scp -q -o BatchMode=yes /srv/workspace/flychess/src/Stockfish/src/stockfish $S:$D/n16x86 || true
scp -q -o BatchMode=yes /tmp/sf8src/src/stockfish $S:$D/sf8x86 || true
ssh -o BatchMode=yes $S "file $D/n16x86 | cut -d: -f2 | cut -c1-40; file $D/sf8x86 | cut -d: -f2 | cut -c1-40"
