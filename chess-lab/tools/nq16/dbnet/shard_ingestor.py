#!/usr/bin/env python3
"""Shard ingestor: round-robins the 16 per-spark JSONL files into the DB.
Reads each file's NEW lines, batch-inserts labels + deepest, truncates the file.
Parallel-friendly: single writer, sequential files, no contention."""
import json
import os
import sqlite3
import time

SHARD = "/srv/workspace/chess-active/shards"
DB = "/srv/workspace/chess-active/store/games.db"
POLL = 30
NODES = [f"spark{i}" for i in "0123456789abcdef"]


def ingest_file(con, path, node):
    if not os.path.exists(path):
        return 0
    lines = []
    with open(path) as f:
        for line in f:
            lines.append(line)
    if not lines:
        return 0
    labels = []
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        fen = r["fen"]
        for d, cp in r["s"].items():
            labels.append((fen, "sf17", int(d), int(cp), node, ""))
    con.executemany(
        "insert or replace into labels(fen, engine, depth, cp, by_node, labeled_at)"
        " values (?,?,?,?,?, datetime('now'))", labels)
    # deepest per fen from this batch
    best = {}
    for fen, _, d, cp, _, _ in labels:
        d = int(d)
        if fen not in best or d > best[fen][0]:
            best[fen] = (d, cp)
    con.executemany(
        "insert into deepest(fen, engine, depth, cp, by_node, labeled_at)"
        " values (?,?,?,?,?, datetime('now'))"
        " on conflict(fen) do update set depth=excluded.depth, cp=excluded.cp,"
        " by_node=excluded.by_node, labeled_at=excluded.labeled_at"
        " where excluded.depth > deepest.depth",
        [(f, "sf17", d, cp, node, "") for f, (d, cp) in best.items()])
    # mark allocations done for these fens
    con.executemany(
        "update allocations set state='done' where fen=? and state='open'",
        [(f,) for f in best])
    # truncate the shard file (worker appends, we consume)
    with open(path, "w"):
        pass
    return len(lines)


def main():
    con = sqlite3.connect(DB, timeout=120)
    con.execute("pragma journal_mode=wal")
    con.execute("pragma synchronous=NORMAL")
    os.makedirs(SHARD, exist_ok=True)
    while True:
        total = 0
        for node in NODES:
            try:
                total += ingest_file(con, f"{SHARD}/{node}.jsonl", node)
            except Exception as e:
                print(f"{node}: {e}", flush=True)
        con.commit()
        if total:
            print(f"{time.strftime('%H:%M:%S')} ingested {total:,} positions",
                  flush=True)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
