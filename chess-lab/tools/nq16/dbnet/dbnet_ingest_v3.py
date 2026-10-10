#!/usr/bin/env python3
"""Ingest baseline miniature rows (fen, uci, gid, ply) into games.db.positions.
No label seeding — labeling happens via the dbnet service/worker."""
import sqlite3
import sys

DB = "/srv/workspace/chess-active/store/games.db"
ROWS = sys.argv[1]

con = sqlite3.connect(DB)
con.execute("pragma journal_mode=wal")
con.execute("pragma synchronous=OFF")
con.executescript("""
create table if not exists positions(seq integer primary key, fen text unique,
    kind text, ref text);
create table if not exists labels(fen text, engine text, depth integer, cp integer,
    by_node text, labeled_at text, primary key(fen, engine, depth));
""")
n = 0
batch = []
for line in open(ROWS):
    p = line.rstrip("\n").split("\t")
    if len(p) < 4:
        continue
    batch.append((p[0], "mini_game", f"{p[2]}:{p[3]}"))
    n += 1
    if len(batch) >= 100000:
        con.executemany("insert or ignore into positions(fen, kind, ref) values (?,?,?)",
                        batch)
        batch = []
if batch:
    con.executemany("insert or ignore into positions(fen, kind, ref) values (?,?,?)",
                    batch)
con.commit()
tot = con.execute("select count(*) from positions").fetchone()[0]
print(f"ingested {n:,} rows read; positions now {tot:,}")
