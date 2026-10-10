"""D20 resume distributor: positions from the remaining list, evaluated by
spark workers, appended to the existing shard outputs. Resume-aware.

Box side: splits remaining positions into per-spark job files, pushes them,
launches spark workers (16 procs each, depth 20), collects outputs.
"""
import os, sys, subprocess, math

RAID = "/mnt/cold-raid6/chess-audit"
SHARDS = f"{RAID}/depth_db/shards"
REM = f"{RAID}/depth_db/remaining.txt"
JOBS = f"{RAID}/depth_db/jobs"
SPARKS = "spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkd sparke sparkf".split()

os.makedirs(JOBS, exist_ok=True)

# done set from existing shard outputs
done = set()
for f in sorted(os.listdir(SHARDS)):
    if not f.endswith(".tsv"):
        continue
    for line in open(f"{SHARDS}/{f}", errors="ignore"):
        p = line.rstrip("\n").split("\t")
        if p:
            done.add(p[0])
print(f"done positions: {len(done):,}", flush=True)

todo = []
for line in open(REM, errors="ignore"):
    fen = line.strip()
    if fen and fen not in done:
        todo.append(fen)
print(f"remaining: {len(todo):,}", flush=True)

CHUNK = math.ceil(len(todo) / len(SPARKS))
for i, s in enumerate(SPARKS):
    part = todo[i * CHUNK:(i + 1) * CHUNK]
    if not part:
        continue
    with open(f"{JOBS}/d20job_{s}.txt", "w") as f:
        f.write("\n".join(part) + "\n")
    subprocess.run(["scp", "-q", "-o", "ConnectTimeout=15",
                    f"{JOBS}/d20job_{s}.txt", f"{s}:~/d20job.txt"], check=False)
    print(f"{s}: {len(part):,} positions", flush=True)
print("JOBS DISTRIBUTED", flush=True)
