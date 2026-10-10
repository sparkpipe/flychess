#!/usr/bin/env python3
"""One-shot: create already-expired leases for every unlabeled position so the
retry-serving in allocate() hands them out on the next call."""
import sqlite3
import time

DB = "/srv/workspace/chess-active/store/games.db"
con = sqlite3.connect(DB)
con.execute("pragma journal_mode=wal")
now = time.time()
rows = con.execute(
    "select p.fen from positions p where not exists("
    "select 1 from labels l where l.fen=p.fen and l.engine='sf17' and l.depth=12)"
    " and not exists(select 1 from allocations a where a.fen=p.fen"
    " and a.engine='sf17' and a.depth=12 and a.state='open')").fetchall()
print(f"orphans to backfill: {len(rows):,}")
con.executemany(
    "insert into allocations(fen, engine, depth, node, state, allocated_at, deadline)"
    " values (?, 'sf17', 12, 'backfill', 'open', ?, ?)",
    [(r[0], now - 7200, now - 3600) for r in rows])  # already expired
con.commit()
print("backfill leases created (pre-expired)")
