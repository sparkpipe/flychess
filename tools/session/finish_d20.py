#!/usr/bin/env python3
"""One-shot: finish the last d20 stragglers on the box. Finds positions
without a d20 label directly (single slow query, no /next timeout), then
evals each with the SF pool and POSTs results through the service."""
import json
import os
import select
import subprocess
import sys
import time
import urllib.request

DB = "/srv/workspace/chess-active/store/games.db"
SF = "/srv/workspace/flychess/src/sf17/src/stockfish"
SVC = "http://127.0.0.1:8125/result"
CORES = 8

import sqlite3
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
print("scanning for missing d20 labels (slow, ~minutes)...", flush=True)
t0 = time.time()
missing = [r[0] for r in con.execute(
    "select p.fen from positions p left join labels l"
    " on l.fen = p.fen and l.engine='sf17' and l.depth=20"
    " where l.fen is null order by p.seq")]
print(f"missing: {len(missing)} (scan {time.time()-t0:.0f}s)", flush=True)


class Child:
    def __init__(self):
        self.p = subprocess.Popen([SF], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
        os.set_blocking(self.p.stdout.fileno(), False)
        self.p.stdin.write(b"uci\nsetoption name Threads value 1\nisready\n")
        self.p.stdin.flush()
        self.buf = b""
        self.scores = {}
        self.cur = None

    def fd(self):
        return self.p.stdout.fileno()

    def alive(self):
        return self.p.poll() is None

    def send(self, fen):
        self.p.stdin.write(f"position fen {fen}\ngo depth 20\n".encode())
        self.p.stdin.flush()


def parse(line):
    t = line.split()
    d = None
    for i, x in enumerate(t):
        if x == "depth":
            try:
                d = int(t[i + 1])
            except (ValueError, IndexError):
                return None
            break
    if d is None:
        return None
    for i, x in enumerate(t):
        if i > 0 and t[i - 1] == "score" and x in ("cp", "mate") and i + 1 < len(t):
            try:
                v = int(t[i + 1])
            except ValueError:
                return None
            val = ((10000 + min(abs(v), 900)) * (1 if v > 0 else -1)
                   if x == "mate" else v)
            return d, val
    return None


kids = [Child() for _ in range(CORES)]
pending = list(missing)
rows_out = []
n = 0
while pending or any(k.cur for k in kids):
    for k in kids:
        if not k.alive():
            k = Child()
        if k.cur is None and pending:
            k.cur = pending.pop(0)
            k.scores = {}
            k.send(k.cur)
    fds = {k.fd(): i for i, k in enumerate(kids) if k.alive() and k.cur}
    if not fds:
        break
    r, _, _ = select.select(list(fds), [], [], 1.0)
    for fd in r:
        i = fds[fd]
        k = kids[i]
        try:
            data = os.read(fd, 65536)
        except BlockingIOError:
            continue
        if not data:
            if k.cur:
                pending.append(k.cur)
                k.cur = None
            continue
        k.buf += data
        while b"\n" in k.buf:
            raw, k.buf = k.buf.split(b"\n", 1)
            line = raw.decode(errors="replace")
            ds = parse(line)
            if ds:
                k.scores[ds[0]] = ds[1]
            elif line.startswith("bestmove") and k.cur:
                stm = 1 if k.cur.split()[1] == "w" else -1
                for d, v in k.scores.items():
                    rows_out.append({"i": 0, "fen": k.cur, "depth": d,
                                     "cp": v * stm, "node": "finisher"})
                n += 1
                k.cur = None
    if len(rows_out) >= 4000:
        req = urllib.request.Request(
            SVC, data=json.dumps({"rows": rows_out}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=300).read()
        rows_out = []
        print(f"{time.strftime('%H:%M:%S')} done {n}/{len(missing)}", flush=True)
if rows_out:
    req = urllib.request.Request(
        SVC, data=json.dumps({"rows": rows_out}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=300).read()
print(f"FINISHED {n} positions, all posted", flush=True)
