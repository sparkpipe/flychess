"""Load provenance TSV into the store DB (extnvme active copy).

Tables:
  positions(fen TEXT PRIMARY KEY, source TEXT)          -- every position known
  train_provenance(fen, source, src_key, engine_set, expert, split, reason)
    index on (engine_set, expert)
"""
import sqlite3, sys, time

DB = "/extnvme/store/games.db"
TSV = "/extnvme/active/provenance.tsv"

con = sqlite3.connect(DB)
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS positions (fen TEXT PRIMARY KEY, source TEXT) WITHOUT ROWID")
cur.execute("""CREATE TABLE IF NOT EXISTS train_provenance (
  fen TEXT, source TEXT, src_key TEXT, engine_set TEXT,
  expert TEXT, split TEXT, reason TEXT)""")
cur.execute("CREATE INDEX IF NOT EXISTS tp_set_expert ON train_provenance(engine_set, expert)")
con.commit()

buf = []
pbuf = {}
t0 = time.time()
n = 0
for line in open(TSV):
    p = line.rstrip("\n").split("\t")
    if len(p) < 7:
        continue
    buf.append(tuple(p))
    pbuf.setdefault(p[0], p[1])
    if len(buf) >= 100000:
        cur.executemany("INSERT OR REPLACE INTO train_provenance VALUES (?,?,?,?,?,?,?)", buf)
        cur.executemany("INSERT OR IGNORE INTO positions VALUES (?,?)", list(pbuf.items()))
        con.commit()
        buf.clear(); pbuf.clear()
    n += 1
    if n % 5000000 == 0:
        print(f"{n:,} ({time.time()-t0:.0f}s)", flush=True)
if buf:
    cur.executemany("INSERT OR REPLACE INTO train_provenance VALUES (?,?,?,?,?,?,?)", buf)
    cur.executemany("INSERT OR IGNORE INTO positions VALUES (?,?)", list(pbuf.items()))
    con.commit()
print(f"DONE {n:,} rows in {time.time()-t0:.0f}s")
cur.execute("SELECT engine_set, COUNT(*) FROM train_provenance GROUP BY engine_set")
for r in cur.fetchall():
    print(r)
