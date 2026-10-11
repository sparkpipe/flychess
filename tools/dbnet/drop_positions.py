import sqlite3

c = sqlite3.connect("store/games.db")
c.execute("drop table if exists positions")
c.commit()
n = c.execute("select count(*) from sqlite_master where name='positions'").fetchone()[0]
print("positions dropped:", n == 0)
