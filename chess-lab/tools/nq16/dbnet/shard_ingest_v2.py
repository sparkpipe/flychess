#!/usr/bin/env python3
"""One-shot shard ingestor (replaces the loop-inside-script version).
Called by the puller after each pull cycle. Single connection, WAL, retries."""
import glob
import json
import os
import sqlite3
import sys
import time

SHARD = "/srv/workspace/chess-active/shards"
DB = "/srv/workspace/chess-active/store/games.db"

con = None
for attempt in range(30):
    try:
        con = sqlite3.connect(DB, timeout=60)
        con.execute("pragma journal_mode=wal")
        con.execute("pragma busy_timeout=60000")
        con.execute("begin immediate")
        break
    except sqlite3.OperationalError:
        time.sleep(2)
if con is None:
    print("could not get write lock after 60s", flush=True)
    sys.exit(1)

total = 0
ts = time.strftime("%Y-%m-%dT%H:%M:%S")
for path in sorted(glob.glob(f"{SHARD}/*_pull.jsonl")):
    try:
        lines = open(path).readlines()
    except FileNotFoundError:
        continue
    if not lines:
        os.remove(path)
        continue
    node = os.path.basename(path).replace("_pull.jsonl", "")
    labels = []
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        fen = r["fen"]
        for d, cp in r["s"].items():
            labels.append((fen, "sf17", int(d), int(cp), node, ts))
    if not labels:
        os.remove(path)
        continue
    con.executemany(
        "insert or replace into labels(fen, engine, depth, cp, by_node, labeled_at)"
        " values (?,?,?,?,?,?)", labels)
    best = {}
    for fen, _, d, cp, _, _ in labels:
        d = int(d)
        if fen not in best or d > best[fen][0]:
            best[fen] = (d, cp)
    con.executemany(
        "insert into deepest(fen, engine, depth, cp, by_node, labeled_at)"
        " values (?,?,?,?,?,?)"
        " on conflict(fen) do update set depth=excluded.depth, cp=excluded.cp,"
        " by_node=excluded.by_node, labeled_at=excluded.labeled_at"
        " where excluded.depth > deepest.depth",
        [(f, "sf17", d, cp, node, ts) for f, (d, cp) in best.items()])
    con.executemany(
        "update allocations set state='done' where fen=? and state='open'",
        [(f,) for f in best])
    os.remove(path)
    total += len(lines)
con.commit()
con.close()
if total:
    print(f"{time.strftime('%H:%M:%S')} ingested {total:,} positions", flush=True)
