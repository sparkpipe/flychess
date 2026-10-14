import sqlite3

c = sqlite3.connect("/srv/workspace/chess-active/store/games.db",
                    timeout=300)
r = c.execute("pragma wal_checkpoint(TRUNCATE)").fetchone()
print("checkpoint:", r)
c.close()
