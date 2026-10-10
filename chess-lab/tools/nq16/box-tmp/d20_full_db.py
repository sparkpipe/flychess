"""DB-WIDE D20 PASS — every eval-bearing ply in the store gets d20.

Population: all distinct board keys in plies with cp IS NOT NULL (d12 labels)
minus keys already having d20 (depth_db shards). Output: fen lists split 16
ways, Mac-relayed to sparks, d20_worker (SF17, all-depths capture).
Result shards collected -> d20 table in the store (key -> cp@d1..d20).
Enables d12..d20 segment extraction per operator plan.
"""
import sqlite3, os

DB = "/mnt/cold-raid6/chess-audit/wp_fit/store/games.db"
OUT = "/extnvme/backlog_eval/d20db"
os.makedirs(OUT, exist_ok=True)

# keys with d20 already: box shards + collected spark shards
import glob
have = set()
for f in glob.glob("/mnt/cold-raid6/chess-audit/depth_db/sparks/*/*.tsv.*") + \
         glob.glob("/mnt/cold-raid6/chess-audit/depth_db/shards/w*.tsv") + \
         glob.glob(OUT + "/collected/*.tsv"):
    try:
        for line in open(f, errors="replace"):
            have.add(line.split("\t")[0].split(" ")[0])
    except Exception:
        pass
print(f"existing d20 keys: {len(have):,}", flush=True)

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
out = open(f"{OUT}/todo.txt", "w", buffering=1 << 20)
n = 0
for (fen,) in con.execute(
        "SELECT fen FROM plies WHERE cp IS NOT NULL"):
    key = fen.split(" ")[0]
    if key in have:
        continue
    out.write(fen + "\n")
    n += 1
out.close()
print(f"need d20: {n:,}", flush=True)
os.system(f"cd {OUT} && split -n l/16 -d --additional-suffix=.todo todo.txt tdb_ && ls tdb_*.todo | wc -l")
