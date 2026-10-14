import sqlite3
import time

c = sqlite3.connect("/srv/workspace/chess-active/store/games.db", timeout=120)
c.execute("pragma journal_mode=wal")
c.execute("pragma busy_timeout=60000")
# expire ALL open d20 leases (fleet-era ghosts) and reset the d20 cursor
c.execute("delete from allocations where state='open'")
c.execute("update meta set value=0 where key='label_cursor_sf17_20'")
c.commit()
o = c.execute("select count(*) from allocations where state='open'").fetchone()[0]
v = c.execute("select value from meta where key='label_cursor_sf17_20'").fetchone()[0]
print(f"open leases: {o:,}; d20 cursor: {v}")
