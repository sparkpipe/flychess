"""Split the tactics bin by expert domain (one full pass, 40B records)."""
import sys
import os

sys.path.insert(0, "/home/spec/chess-lab/tools")
from audit_packer import unpack_sfen
from score_experts import domain

SRC = "/mnt/cold-raid6/chess-audit/expert_bins_both/tactics.bin"
OUT = "/mnt/cold-raid6/chess-audit/tactics_split"
os.makedirs(OUT, exist_ok=True)
files = {}
n = 0
with open(SRC, "rb") as f:
    while True:
        r = f.read(40)
        if len(r) < 40:
            break
        b, hm, fm = unpack_sfen(r[:32])
        d = domain(b.fen())
        if d not in files:
            files[d] = open(os.path.join(OUT, d + ".bin"), "wb")
        files[d].write(r)
        n += 1
        if n % 2000000 == 0:
            print(n, flush=True)
for fh in files.values():
    fh.close()
print("SPLIT DONE:", n)
