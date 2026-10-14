import sqlite3

c = sqlite3.connect("file:/srv/workspace/chess-active/store/games.db?mode=ro",
                    uri=True)
for k, v in c.execute("select key, value from meta where key like 'label_cursor%'").fetchall():
    print(f"{k}: {v}")
pos = c.execute("select max(seq) from positions").fetchone()[0]
print(f"max position seq: {pos}")
