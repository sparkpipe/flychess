#!/usr/bin/env python3
"""Ingest miniature positions (fen rows WITHOUT evals) into games.db.positions,
and seed already-known SF17 d12 labels from the relabel part-file.
idempotent; kind='mini_game', ref='gid:ply'."""
import sqlite3
import sys

DB = "/srv/workspace/chess-active/store/games.db"
ROWS = "/srv/workspace/chess-active/miniature_rows.tsv"          # fen, uci, gid, ply
SEED = "/srv/workspace/chess-active/miniature_labels17.tsv.part"  # fen \t cp (stm)

con = sqlite3.connect(DB)
con.execute("pragma journal_mode=wal")
con.execute("pragma synchronous=OFF")
con.executescript("""
create table if not exists positions(seq integer primary key, fen text unique, kind text, ref text);
create table if not exists labels(fen text, engine text, depth integer, cp integer,
    by_node text, labeled_at text, primary key(fen, engine, depth));
""")

n = dup = 0
batch = []
for line in open(ROWS):
    p = line.rstrip("\n").split("\t")
    if len(p) < 4:
        continue
    fen, uci, gid, ply = p[0], p[1], p[2], p[3]
    batch.append((fen, "mini_game", f"{gid}:{ply}"))
    n += 1
    if len(batch) >= 100000:
        con.executemany("insert or ignore into positions(fen, kind, ref) values (?,?,?)", batch)
        batch = []
if batch:
    con.executemany("insert or ignore into positions(fen, kind, ref) values (?,?,?)", batch)
con.commit()
print(f"positions ingested: {n:,} rows read")

m = 0
batch = []
for line in open(SEED):
    fen, cp = line.rstrip("\n").split("\t")
    batch.append((fen, "sf17", 12, int(cp), "box-relabel17",
                  "2026-10-07T08:00:00"))
    m += 1
    if len(batch) >= 100000:
        con.executemany("insert or ignore into labels values (?,?,?,?,?,?)", batch)
        batch = []
if batch:
    con.executemany("insert or ignore into labels values (?,?,?,?,?,?)", batch)
con.commit()
print(f"labels seeded: {m:,} sf17-d12 rows")
tot = con.execute("select count(*) from positions").fetchone()[0]
lab = con.execute("select count(*) from labels where engine='sf17' and depth=12").fetchone()[0]
print(f"DB now: positions={tot:,}  sf17-d12 labels={lab:,}  unlabeled~={tot - lab:,}")
