#!/usr/bin/env python3
"""D20 sweep service — the operator protocol, nothing else.

One global cursor: the seq of the next position WITHOUT a d20 eval.
GET  /next?n=K  -> {"items": [{"i": seq, "fen": fen}, ...]}   (empty + done:true when the sweep is complete)
POST /result    {"rows": [{"i", "fen", "depth", "cp"}, ...]}  -> fraction-of-a-ms PK updates

A mutex serializes cursor + writes. No leases, no lockfiles.
Cursor persists to meta every 5000 handouts so a restart skips ahead."""
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KIND = os.environ.get("SWEEP_KIND", "")
DB = "/srv/workspace/chess-active/store/games.db"
CURSOR_KEY = "d20_cursor"
SAVE_EVERY = 5000

con = sqlite3.connect(DB, check_same_thread=False, timeout=60)
con.execute("pragma journal_mode=wal")
con.execute("pragma synchronous=NORMAL")
mu = threading.Lock()


def load_cursor():
    r = con.execute("select value from meta where key=?", (CURSOR_KEY,)).fetchone()
    return r[0] if r else 0


cursor = load_cursor()
since_save = 0
done = False


def next_batch(n):
    """Advance the global cursor to the next positions lacking a d20 label.
    Wraps to 0 at the end and keeps skipping labeled rows; a full circle
    with nothing found marks the sweep done. Positions re-handed after a
    wrap are idempotent (insert-or-replace)."""
    global cursor, since_save, done
    with mu:
        if done:
            return [], True
        items = []
        while len(items) < n:
            q = ("select p.seq, p.fen from positions p"
                 " left join labels l on l.fen = p.fen"
                 "   and l.engine=? and l.depth=20"
                 " where p.seq > ? and l.fen is null"
                 + (" and p.kind=?" if KIND else "")
                 + " order by p.seq limit ?")
            params = ("sf17", cursor, n - len(items))
            if KIND:
                params = ("sf17", cursor, KIND, n - len(items))
            rows = con.execute(q, params).fetchall()
if rows:
                items += [{"i": s, "fen": f} for s, f in rows]
                cursor = rows[-1][0]
            elif cursor > 0:
                cursor = 0
            else:
                break
        if not items:
            done = True
            con.execute("insert or replace into meta values (?,?)",
                        (CURSOR_KEY, cursor))
            con.commit()
            return [], True
        since_save += len(items)
        if since_save >= SAVE_EVERY:
            con.execute("insert or replace into meta values (?,?)",
                        (CURSOR_KEY, cursor))
            con.commit()
            since_save = 0
        return items, False


def store(rows):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    with mu:
        con.executemany(
            "insert or replace into labels(fen, engine, depth, cp, by_node,"
            " labeled_at) values (?,?,?,?,?,?)",
            [(r["fen"], "sf17", int(r["depth"]), int(r["cp"]),
              r.get("node", "?"), ts) for r in rows])
        con.executemany(
            "insert into deepest(fen, engine, depth, cp, by_node, labeled_at)"
            " values (?,?,?,?,?,?) on conflict(fen) do update set"
            " depth=excluded.depth, cp=excluded.cp, by_node=excluded.by_node,"
            " labeled_at=excluded.labeled_at where excluded.depth > deepest.depth",
            [(r["fen"], "sf17", int(r["depth"]), int(r["cp"]),
              r.get("node", "?"), ts)
             for r in rows if int(r["depth"]) >= 20])
        con.commit()
    return len(rows)


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
        if self.path.startswith("/next"):
            n = 1
            if "n=" in self.path:
                try:
                    n = max(1, min(int(self.path.split("n=")[1].split("&")[0]), 20000))
                except ValueError:
                    pass
            items, fin = next_batch(n)
            return self._json(200, {"items": items, "done": fin})
        self._json(404, {"err": "no"})

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(ln))
        except Exception:
            return self._json(400, {"err": "bad json"})
        if self.path == "/result":
            return self._json(200, {"stored": store(body.get("rows", []))})
        self._json(404, {"err": "no"})


if __name__ == "__main__":
    print(f"d20 sweep service on 0.0.0.0:8125 (cursor={cursor})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8125), H).serve_forever()
