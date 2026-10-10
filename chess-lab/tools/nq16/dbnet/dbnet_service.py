#!/usr/bin/env python3
"""Network-updatable labeling service over games.db (stdlib only).
Serial-cursor allocation: no searching for unlabeled rows — the cursor walks
positions in insertion order; expired leases are re-served from a small retry set.
POST /allocate {node, engine, depth, n} -> {allocated: N, fens: [...]}
POST /submit   {node, results: [{fen, cp}]} -> {stored: N}
GET  /stats | GET /position/<fen>
Bind 127.0.0.1 for debug; sparks use the same HTTP path with the LAN address."""
import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DB = "/srv/workspace/chess-active/store/games.db"
HOST = "0.0.0.0"  # spark access
PORT = 8123
LEASE_S = 600
N_MAX = 20000

con = sqlite3.connect(DB, check_same_thread=False, timeout=60)
con.execute("pragma journal_mode=wal")
con.execute("pragma synchronous=NORMAL")
lock = threading.Lock()


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
    """)
    # covering index: depth-keyed exports/completeness never table-scan
    con.execute("create index if not exists idx_labels_engdep"
                " on labels(engine, depth, fen, cp)")
    con.commit()
    # one-time cursor init (can take a minute on first boot; done before serving)
    get_cursor("sf17", 12)


def cursor_key(engine, depth):
    return f"label_cursor_{engine}_{depth}"


def get_cursor(engine, depth):
    r = con.execute("select value from meta where key=?",
                    (cursor_key(engine, depth),)).fetchone()
    if r is None:
        # initialize: skip everything already labeled (seed), assuming in-order seeding
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
        # re-serve expired leases (retry gap) — re-lease ONLY the n handed out;
        # unserved expired rows stay in place for the next allocate
        expired = [r[0] for r in con.execute(
            "select distinct fen from allocations where state='open' and deadline < ?"
            " limit ?", (now, n))]
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
                "select seq, fen from positions where seq > ? order by seq limit ?",
                (cur, n - len(fens))).fetchall()
            if rows:
                cur = rows[-1][0]
                fens += [r[1] for r in rows]
                con.execute("insert or replace into meta values (?,?)",
                            (cursor_key(engine, depth), cur))
        con.executemany(
            "insert into allocations(fen, engine, depth, node, state, allocated_at, deadline)"
            " values (?,?,?,?, 'open', ?, ?)",
            [(f, engine, depth, node, now, now + LEASE_S) for f in fens])
        con.commit()
    return fens


def submit(node, results, engine="sf17", depth=12):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    with lock:
        rows = []
        deep = []
        for r in results:
            # v2 workers record a DEPTH TRAJECTORY (13..D) per position, so the
            # row's own depth wins when present
            d = int(r.get("depth", depth))
            rows.append((r["fen"], engine, d, int(r["cp"]), node, ts))
            deep.append((r["fen"], engine, d, int(r["cp"]), node, ts))
        con.executemany(
            "insert or replace into labels(fen, engine, depth, cp, by_node, labeled_at)"
            " values (?,?,?,?,?,?)", rows)
        # deepest-known eval per fen (for variable-depth training): upsert only
        # when this row's depth exceeds what is stored
        con.executemany(
            "insert into deepest(fen, engine, depth, cp, by_node, labeled_at)"
            " values (?,?,?,?,?,?) on conflict(fen) do update set"
            " engine=excluded.engine, depth=excluded.depth, cp=excluded.cp,"
            " by_node=excluded.by_node, labeled_at=excluded.labeled_at"
            " where excluded.depth > deepest.depth", deep)
        con.executemany(
            "update allocations set state='done' where fen=? and state='open'"
            " and engine=? and depth=?",
            [(r["fen"], engine, depth) for r in results])
        con.commit()
    return len(results)


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
    return {"positions": pos, "deepest": dep,
            "labels": [{"engine": e, "depth": d, "n": n} for e, d, n in labs],
            "open_leases": open_a, "cursors": curs}


def position(fen):
    with lock:
        labels = con.execute(
            "select engine, depth, cp, by_node, labeled_at from labels where fen=?",
            (fen,)).fetchall()
        pos = con.execute("select kind, ref from positions where fen=?", (fen,)).fetchall()
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
            k = submit(body.get("node", "?"), body.get("results", []),
                       body.get("engine", "sf17"), int(body.get("depth", 12)))
            return self._json(200, {"stored": k})
        self._json(404, {"err": "no"})


if __name__ == "__main__":
    init()
    print(f"labeling service on {HOST}:{PORT}, db={DB}", flush=True)
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
