#!/bin/bash
# spark2 data-generation pilot (operator: "start with one spark")
# Stages gen script, runs 50 games, reports output size + timing
cd /home/spark2/extnvme/phase-moe
sed -i 's|^SF = .*|SF = "/home/spark2/extnvme/phase-moe/sf/src/stockfish"|' gen_smoke_data.py
sed -i 's|^OUT = .*|OUT = "/home/spark2/extnvme/phase-moe/spark_data.bin"|' gen_smoke_data.py
GAMES=50 DEPTH=8 timeout 600 python3 gen_smoke_data.py
echo "SIZE: $(stat -c %s spark_data.bin 2>/dev/null || echo 0)"
echo "SPARK-PILOT-DONE"
