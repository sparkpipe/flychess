#!/usr/bin/env python3
"""Local shard ingestor: reads box-side pulled JSONL files, batch-inserts to DB."""
import glob
import json
import os
import sqlite3
import time

SHARD = "/srv/workspace/chess-active/shards"
DB = "/srv/workspace/chess-active/store/games.db"


def ingest_file(con, path):
    lines = open(path).readlines()
    if not lines:
        os.remove(path)
        return 0
    node = os.path.basename(path).replace("_pull.jsonl", "").replace(".jsonl", "")
    labels = []
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        fen = r["fen"]
        for d, cp in r["s"].items():
            labels.append((fen, "sf17", int(d), int(cp), node, ""))
    if not labels:
        os.remove(path)
        return 0
    con.executemany(
        "insert or replace into labels(fen, engine, depth, cp, by_node, labeled_at)"
        " values (?,?,?,?,?, datetime('now'))", labels)
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
        [(f, "sf17", d, cp, node) for f, (d, cp) in best.items()])
    con.executemany(
        "update allocations set state='done' where fen=? and state='open'",
        [(f,) for f in best])
    con.commit()
    os.remove(path)
    return len(lines)


def main():
    con = sqlite3.connect(DB, timeout=120)
    con.execute("pragma journal_mode=wal")
    con.execute("pragma synchronous=NORMAL")
    total = 0
    for path in sorted(glob.glob(f"{SHARD}/*_pull.jsonl")):
        try:
            total += ingest_file(con, path)
        except Exception as e:
            print(f"{path}: {e}", flush=True)
    if total:
        print(f"{time.strftime('%H:%M:%S')} ingested {total:,} positions", flush=True)


if __name__ == "__main__":
    main()
