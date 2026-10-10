#!/bin/bash
# Download 6-piece WDL tablebases from lichess (rook endings first)
# Runs on spark2, writes to local NVMe
BASE=https://tablebase.lichess.ovh/tables/standard/6-wdl
DEST=~/extnvme/syzygy6
mkdir -p $DEST
cd $DEST

# get the full file listing
curl -s "$BASE/" | grep -o 'href="[^"]*"' | sed 's/href="//;s/"//' | grep "\.rtbw$" > /tmp/rtbw_files.txt

# sort: rook endings first (most practical), then queen, then pawn-heavy
grep "^KR" /tmp/rtbw_files.txt > /tmp/priority.txt
grep "^KQ" /tmp/rtbw_files.txt >> /tmp/priority.txt
grep "^KP" /tmp/rtbw_files.txt >> /tmp/priority.txt
grep -v "^K[RPQ]" /tmp/rtbw_files.txt >> /tmp/priority.txt

echo "downloading $(wc -l < /tmp/priority.txt) WDL files (rook first)..."
for F in $(cat /tmp/priority.txt); do
    if [ ! -f "$F" ]; then
        curl -s -o "$F" "$BASE/$F" &
        # limit to 4 parallel downloads
        while [ $(jobs -r | wc -l) -ge 4 ]; do
            sleep 0.5
        done
    fi
done
wait
echo "6-MAN-WDL-DOWNLOAD-COMPLETE: $(ls *.rtbw 2>/dev/null | wc -l) files, $(du -sh . | cut -f1)"
