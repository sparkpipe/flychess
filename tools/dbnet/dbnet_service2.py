#!/usr/bin/env python3
"""Labeling service — operator design, exactly as specified:
  - Workers submit evals via HTTP /submit
  - Service appends each node's batch to shards/{node}.jsonl (append-only,
    one file per node, local disk, zero cross-machine file ops)
  - A serial ingestor (same process) round-robins the 16 files and
    batch-inserts into the DB. One writer, one reader, no contention.

POST /allocate {node, engine, depth, n} -> {allocated: N, fens: [...]}
POST /submit   {node, results: [{fen, cp, depth}]} -> {stored: N}
GET  /stats | /position/<fen>"""
import glob
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DB = "/srv/workspace/chess-active/store/games.db"
SHARD = "/srv/workspace/chess-active/shards"
HOST = "0.0.0.0"
PORT = 8123
LEASE_S = 600
N_MAX = 20000
INGEST_POLL = 5  # seconds between round-robin passes

con = sqlite3.connect(DB, check_same_thread=False, timeout=60)
con.execute("pragma journal_mode=wal")
con.execute("pragma synchronous=NORMAL")
lock = threading.Lock()
os.makedirs(SHARD, exist_ok=True)


def init():
    con.executescript("""
    create table if not exists positions(seq integer primary key, fen text unique,
        kind text, ref text);
    create table if not exists labels(fen text, engine text, depth integer, cp integer,
        by_node text, labeled_at text, primary key(fen, engine, depth));
    create table if not exists allocations(id integer primary key autoincrement, fen text,
        engine text, depth integer, node text, state text, allocated_at real, deadline real);
    create index if not exists idx_alloc_open on allocations(state, deadline);
    create table if not exists meta(key text primary key, value integer);
    create table if not exists deepest(fen text primary key, engine text,
        depth integer, cp integer, by_node text, labeled_at text);
    create index if not exists idx_labels_engdep on labels(engine, depth, fen, cp);
    """)
    con.commit()
    get_cursor("sf17", 12)
    get_cursor("sf17", 20)


def cursor_key(engine, depth):
    return f"label_cursor_{engine}_{depth}"


def get_cursor(engine, depth):
    r = con.execute("select value from meta where key=?",
                    (cursor_key(engine, depth),)).fetchone()
    if r is None:
        r2 = con.execute(
            "select coalesce(max(p.seq), 0) from positions p join labels l "
            "on l.fen = p.fen and l.engine = ? and l.depth = ?",
            (engine, depth)).fetchone()
        con.execute("insert or replace into meta values (?,?)",
                    (cursor_key(engine, depth), r2[0]))
        con.commit()
        return r2[0]
    return r[0]


def allocate(node, engine, depth, n):
    n = min(int(n), N_MAX)
    now = time.time()
    with lock:
        expired = [r[0] for r in con.execute(
            "select distinct fen from allocations where state='open'"
            " and deadline < ? limit ?", (now, n))]
        fens = expired
        if expired:
            con.execute(
                "update allocations set node=?, deadline=? where state='open'"
                " and deadline < ? and fen in (select fen from ("
                "  select distinct fen from allocations where state='open'"
                "  and deadline < ? limit ?))",
                (node, now + LEASE_S, now, now, n))
        if len(fens) < n:
            cur = get_cursor(engine, depth)
            rows = con.execute(
                "select seq, fen from positions where seq > ? order by seq"
                " limit ?", (cur, n - len(fens))).fetchall()
            if rows:
                cur = rows[-1][0]
                fens += [r[1] for r in rows]
                con.execute("insert or replace into meta values (?,?)",
                            (cursor_key(engine, depth), cur))
        con.executemany(
            "insert into allocations(fen, engine, depth, node, state,"
            " allocated_at, deadline) values (?,?,?,?, 'open', ?, ?)",
            [(f, engine, depth, node, now, now + LEASE_S) for f in fens])
        con.commit()
    return fens


def submit(node, results):
    """Append batch to this node's shard file. Fast, no DB lock."""
    path = f"{SHARD}/{node}.jsonl"
    with open(path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return len(results)


def ingestor():
    """Round-robin the 16 shard files, batch-insert serially into the DB."""
    while True:
        ingested = 0
        for path in sorted(glob.glob(f"{SHARD}/*.jsonl")):
            try:
                with open(path) as f:
                    lines = f.readlines()
            except FileNotFoundError:
                continue
            if not lines:
                continue
            node = os.path.basename(path).replace(".jsonl", "")
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            labels = []
            best = {}
            for line in lines:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fen = r["fen"]
                d = int(r.get("depth", 12))
                cp = int(r["cp"])
                labels.append((fen, "sf17", d, cp, node, ts))
                if fen not in best or d > best[fen][0]:
                    best[fen] = (d, cp)
            if not labels:
                # still truncate so we don't re-read garbage
                with open(path, "w"):
                    pass
                continue
            with lock:
                con.executemany(
                    "insert or replace into labels(fen, engine, depth, cp,"
                    " by_node, labeled_at) values (?,?,?,?,?,?)", labels)
                con.executemany(
                    "insert into deepest(fen, engine, depth, cp, by_node,"
                    " labeled_at) values (?,?,?,?,?,?)"
                    " on conflict(fen) do update set depth=excluded.depth,"
                    " cp=excluded.cp, by_node=excluded.by_node,"
                    " labeled_at=excluded.labeled_at"
                    " where excluded.depth > deepest.depth",
                    [(f, "sf17", d, cp, node, ts)
                     for f, (d, cp) in best.items()])
                con.executemany(
                    "update allocations set state='done' where fen=?"
                    " and state='open'", [(f,) for f in best])
                con.commit()
            # safe: same process, we hold the only write handle pattern
            with open(path, "w"):
                pass
            ingested += len(lines)
        if ingested:
            print(f"{time.strftime('%H:%M:%S')} ingested {ingested:,} rows",
                  flush=True)
        time.sleep(INGEST_POLL)


def stats():
    with lock:
        pos = con.execute("select count(*) from positions").fetchone()[0]
        dep = con.execute("select count(*) from deepest").fetchone()[0]
        labs = con.execute(
            "select engine, depth, count(*) from labels"
            " group by engine, depth").fetchall()
        open_a = con.execute(
            "select count(*) from allocations where state='open'").fetchone()[0]
        curs = dict(con.execute("select key, value from meta").fetchall())
    try:
        pending = sum(1 for _ in open(
            f"{SHARD}/pending_count", "r")) if os.path.exists(
            f"{SHARD}/pending_count") else 0
    except Exception:
        pending = 0
    return {"positions": pos, "deepest": dep,
            "labels": [{"engine": e, "depth": d, "n": n}
                       for e, d, n in labs],
            "open_leases": open_a, "cursors": curs}


def position(fen):
    with lock:
        labels = con.execute(
            "select engine, depth, cp, by_node, labeled_at from labels"
            " where fen=?", (fen,)).fetchall()
        pos = con.execute(
            "select kind, ref from positions where fen=?", (fen,)).fetchall()
        seg = con.execute(
            "select seg_id, gid, ply, band, expert, role from seg_positions"
            " where fen=? limit 5", (fen,)).fetchall()
    return {"fen": fen, "positions": pos, "labels": labels, "seg_rows": seg}


class H(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/stats":
            return self._json(200, stats())
        if self.path.startswith("/position/"):
            from urllib.parse import unquote
            return self._json(200, position(unquote(self.path[10:]).strip()))
        self._json(404, {"err": "no"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n))
        except Exception:
            return self._json(400, {"err": "bad json"})
        if self.path == "/allocate":
            fens = allocate(body.get("node", "?"), body.get("engine", "sf17"),
                            int(body.get("depth", 12)), body.get("n", 2000))
            return self._json(200, {"allocated": len(fens), "fens": fens})
        if self.path == "/submit":
            k = submit(body.get("node", "?"), body.get("results", []))
            return self._json(200, {"stored": k})
        self._json(404, {"err": "no"})


if __name__ == "__main__":
    init()
    t = threading.Thread(target=ingestor, daemon=True)
    t.start()
    print(f"labeling service on {HOST}:{PORT}, shards={SHARD}", flush=True)
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
