"""CORR SPLIT — evaluate the trained op_even_l1 net on game-only vs
puzzle-only val records separately. Uses the 1-step trainer-validation path
on two extracted sub-bins."""
import struct, os, subprocess, sys, glob, re

SRC = "/mnt/cold-raid6/chess-audit/train23/main/op_even_l1.val.bin"
EVAL_DIR = os.path.expanduser("~/backlog_shards/puzzle_eval")
puzzle_keys = set()
for shard in glob.glob(f"{EVAL_DIR}/w*.tsv"):
    for line in open(shard, errors="replace"):
        puzzle_keys.add(line.split("\t")[0].split(" ")[0])

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
from audit_packer import unpack_sfen

raw = open(SRC, "rb").read()
n = len(raw) // 40
g = open("/tmp/opel1_game.val.bin", "wb")
p = open("/tmp/opel1_puzzle.val.bin", "wb")
ng = np_ = 0
for i in range(n):
    rec = raw[i*40:(i+1)*40]
    try:
        k = unpack_sfen(rec[:32])[0].fen().split(" ")[0]
    except Exception:
        g.write(rec); ng += 1; continue
    if k in puzzle_keys:
        p.write(rec); np_ += 1
    else:
        g.write(rec); ng += 1
g.close(); p.close()
print(f"game-only val: {ng:,}   puzzle-only val: {np_:,}")
