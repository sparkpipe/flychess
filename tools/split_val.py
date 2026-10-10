"""Split each expert bin 95/5 into train/val (40-byte records, line-exact).
Usage: split_val.py <bindir> <traindir> <valdir>
"""
import os
import sys
import random

bindir, traindir, valdir = sys.argv[1], sys.argv[2], sys.argv[3]
random.seed(20260928)
for d in (traindir, valdir):
    os.makedirs(d, exist_ok=True)
REC = 40
for name in sorted(os.listdir(bindir)):
    if not name.endswith(".bin"):
        continue
    src = os.path.join(bindir, name)
    n = os.path.getsize(src) // REC
    keep_val = set(random.sample(range(n), max(1, n // 20)))
    with open(src, "rb") as f, \
         open(os.path.join(traindir, name), "wb") as wt, \
         open(os.path.join(valdir, name), "wb") as wv:
        for i in range(n):
            r = f.read(REC)
            (wv if i in keep_val else wt).write(r)
    print("%s: %d train / %d val" % (name, n - len(keep_val), len(keep_val)))
print("SPLIT-COMPLETE")
