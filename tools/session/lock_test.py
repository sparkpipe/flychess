import sqlite3
import time

c = sqlite3.connect("/srv/workspace/chess-active/store/games.db", timeout=5)
c.execute("pragma journal_mode=wal")
t0 = time.time()
try:
    c.execute("begin immediate")
    c.execute("insert or replace into meta values ('probe', 1)")
    c.commit()
    print(f"write lock: {time.time()-t0:.2f}s")
except Exception as e:
    print(f"write lock FAILED after {time.time()-t0:.2f}s: {e}")
