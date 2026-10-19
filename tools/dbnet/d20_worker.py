#!/usr/bin/env python3
"""d20 sweep worker — cursor protocol: GET /next, POST /result.
One process drives CORES SF children (1 thread each). No leases, no local
state: work not POSTed before a crash is re-handed on the wrap.
Usage: d20_worker.py <node> <cores> <api-url> <sf-binary>"""
import json
import os
import select
import subprocess
import sys
import time
import urllib.request

NODE, CORES, API, SF = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
MAXD = 20
BUF_TARGET = CORES * 150
POST_ROWS = 500


def api(path, body=None):
    for attempt in range(6):
        try:
            if body is None:
                with urllib.request.urlopen(f"{API}{path}", timeout=60) as r:
                    return json.load(r)
            req = urllib.request.Request(f"{API}{path}",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            print(f"api {path} attempt {attempt+1}: {e}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"api {path} failed")


class Child:
    def __init__(self):
        self.p = subprocess.Popen([SF], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        os.set_blocking(self.p.stdout.fileno(), False)
        self.p.stdin.write(b"uci\nsetoption name Hash value 8\n"
                           b"setoption name Threads value 1\nisready\n")
        self.p.stdin.flush()
        self.buf = b""
        self.scores = {}
        self.cur = None
        self.last_out = time.time()

    def fd(self):
        return self.p.stdout.fileno()

    def alive(self):
        return self.p.poll() is None

    def send(self, fen):
        self.p.stdin.write(f"position fen {fen}\ngo depth {MAXD}\n".encode())
        self.p.stdin.flush()

    def kill(self):
        try:
            self.p.kill()
        except Exception:
            pass


def parse_depth_score(line):
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


def main():
    kids = [Child() for _ in range(CORES)]
    pending, out_rows = [], []
    done_flag = False
    n_pos = n_rows = 0
    last_print = time.time()
    while True:
        if not done_flag and len(pending) < BUF_TARGET:
            want = min(BUF_TARGET - len(pending), 5000)
            r = api(f"/next?n={want}")
            pending += r.get("items", [])
            if r.get("done"):
                done_flag = True
        idle = sum(1 for k in kids if k.cur is None)
        if done_flag and not pending and idle == CORES:
            if out_rows:
                api("/result", {"rows": out_rows})
                n_rows += len(out_rows)
                out_rows = []
            break
        for i, k in enumerate(kids):
            if not k.alive():
                kids[i] = Child()
                k = kids[i]
            if k.cur is None and pending:
                k.cur = pending.pop(0)
                k.scores = {}
                k.send(k.cur["fen"])
        fds = {k.fd(): i for i, k in enumerate(kids) if k.alive() and k.cur}
        if not fds:
            time.sleep(0.2)
            continue
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
                k.kill()
                kids[i] = Child()
                continue
            k.buf += data
            while b"\n" in k.buf:
                raw, k.buf = k.buf.split(b"\n", 1)
                line = raw.decode(errors="replace")
                k.last_out = time.time()
                ds = parse_depth_score(line)
                if ds:
                    k.scores[ds[0]] = ds[1]
                elif line.startswith("bestmove"):
                    if k.cur is not None:
                        for d, v in k.scores.items():
                            out_rows.append({"i": k.cur["i"], "fen": k.cur["fen"],
                                             "depth": d, "cp": v, "node": NODE})
                        n_pos += 1
                    k.cur = None
                    k.scores = {}
        if len(out_rows) >= POST_ROWS:
            api("/result", {"rows": out_rows})
            n_rows += len(out_rows)
            out_rows = []
        now = time.time()
        for i, k in enumerate(kids):
            if k.cur and now - k.last_out > 120:
                pending.append(k.cur)
                k.kill()
                kids[i] = Child()
        if now - last_print > 30:
            print(f"{time.strftime('%H:%M:%S')} pos={n_pos:,} rows={n_rows:,} "
                  f"buf={len(pending):,} done_flag={done_flag}", flush=True)
            last_print = now
    print(f"{NODE} SWEEP DONE pos={n_pos:,} rows={n_rows:,}", flush=True)


if __name__ == "__main__":
    main()
