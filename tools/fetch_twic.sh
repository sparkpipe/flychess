#!/bin/bash
# Download the last N TWIC weekly zips (OTB tournament games) and
# concatenate the PGNs into games/twic.pgn
set -u
N=${1:-120}
ALL=${2:-0}
cd /home/spec/chess-lab/games
# find the current issue number from the latest weekly index
LAST=$(curl -sL "https://www.theweekinchess.com/chessnews" | grep -o "twic[0-9]*g\.zip" | head -1 | grep -o "[0-9]*")
if [ -z "$LAST" ]; then LAST=1500; fi
echo "latest issue: $LAST"
if [ "$ALL" = "1" ]; then FIRST=1; else FIRST=$((LAST - N + 1)); fi
for i in $(seq $FIRST $LAST); do
  F="twic${i}g.zip"
  if [ ! -s "$F" ]; then
    curl -sL -o "$F" "https://www.theweekinchess.com/zips/$F"
    sleep 0.3
  fi
  if [ -s "$F" ]; then
    python3 -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); sys.stdout.buffer.write(z.read(z.namelist()[0]))" "$F" >> twic.pgn
  fi
done
echo "twic.pgn bytes: $(stat -c %s twic.pgn)"
