import sqlite3
import time

c = sqlite3.connect("/srv/workspace/chess-active/store/games.db")
c.execute("pragma journal_mode=wal")
t0 = time.time()
c.execute("""insert or ignore into deepest(fen, engine, depth, cp, by_node, labeled_at)
select fen, engine, depth, cp, by_node, labeled_at from labels l
where depth = (select max(depth) from labels l2 where l2.fen = l.fen)""")
c.commit()
n = c.execute("select count(*) from deepest").fetchone()[0]
print(f"deepest backfill: {n:,} rows in {time.time()-t0:.0f}s")
