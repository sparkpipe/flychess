import sqlite3
import time

c = sqlite3.connect("/srv/workspace/chess-active/store/games.db", timeout=600)
c.execute("pragma journal_mode=wal")
c.execute("pragma busy_timeout=300000")
t0 = time.time()
c.execute("delete from allocations where state='open' and deadline < ?",
          (time.time(),))
c.commit()
o = c.execute("select count(*) from allocations where state='open'").fetchone()[0]
t = c.execute("select count(*) from allocations").fetchone()[0]
print(f"purge done in {time.time()-t0:.0f}s; open={o:,} total={t:,}")
