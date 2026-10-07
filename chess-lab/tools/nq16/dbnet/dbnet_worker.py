#!/usr/bin/env python3
"""Labeling worker — same code path for localhost debug and sparks (API is a URL).
C SF17 children (1 fen/core), 1-minute buffer, 30-second replenish, lease-respecting.
Usage: dbnet_worker.py <node> <cores> <depth> <api-url>"""
import json
import os
import select
import subprocess
import sys
import time
import urllib.request

NODE, CORES, DEPTH, API = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
SF17 = "/srv/workspace/flychess/src/sf17/src/stockfish"
BUF_TARGET = CORES * 3000       # ~1 min of work per core at d12
REPLENISH_S = 30


def api(path, body=None, tries=6):
    for attempt in range(tries):
        try:
            if body is None:
                with urllib.request.urlopen(f"{API}{path}", timeout=60) as r:
                    return json.load(r)
            req = urllib.request.Request(f"{API}{path}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            print(f"api {path} attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"api {path} failed after {tries} tries")


class Child:
    def __init__(self):
        self.p = subprocess.Popen([SF17], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        os.set_blocking(self.p.stdout.fileno(), False)
        self.p.stdin.write(b"uci\nisready\n")
        self.p.stdin.flush()
        self.buf = b""
        self.score = None
        self.cur = None
        self.last_out = time.time()

    def fd(self):
        return self.p.stdout.fileno()

    def alive(self):
        return self.p.poll() is None

    def send(self, fen):
        self.p.stdin.write(f"position fen {fen}\ngo depth {DEPTH}\n".encode())
        self.p.stdin.flush()

    def kill(self):
        try:
            self.p.kill()
        except Exception:
            pass


def parse_score(line):
    t = line.split()
    for i, x in enumerate(t):
        if i > 0 and t[i - 1] == "score" and x in ("cp", "mate") and i + 1 < len(t):
            try:
                v = int(t[i + 1])
            except ValueError:
                return None
            if x == "mate":
                return (10000 + min(abs(v), 900)) * (1 if v > 0 else -1)
            return v
    return None


def main():
    kids = [Child() for _ in range(CORES)]
    pending = []
    done_rows = []
    last_replenish = time.time()
    n_sub = 0
    while True:
        # replenish buffer
        if time.time() - last_replenish > REPLENISH_S or len(pending) < CORES:
            if done_rows:
                r = api("/submit", {"node": NODE, "results": done_rows})
                n_sub += r.get("stored", 0)
                done_rows = []
            want = min(BUF_TARGET - len(pending), 20000)
            if want > 0:
                r = api("/allocate", {"node": NODE, "engine": "sf17",
                                      "depth": DEPTH, "n": want})
                pending.extend(r.get("fens", []))
                print(f"{time.strftime('%H:%M:%S')} buffer={len(pending):,} "
                      f"submitted_total={n_sub:,}", flush=True)
            last_replenish = time.time()
            if not pending:
                time.sleep(5)
                continue
        # dispatch
        for i, k in enumerate(kids):
            if not k.alive():
                kids[i] = Child()
            if k.cur is None and pending:
                k.cur = pending.pop(0)
                k.score = None
                k.send(k.cur)
        fds = {k.fd(): i for i, k in enumerate(kids) if k.alive() and k.cur}
        if not fds:
            time.sleep(0.2)
            continue
        try:
            r, _, _ = select.select(list(fds), [], [], 1.0)
        except (OSError, ValueError):
            continue
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
                if line.startswith("info") and " score " in line:
                    s = parse_score(line)
                    if s is not None:
                        k.score = s
                elif line.startswith("bestmove"):
                    if k.cur is not None and k.score is not None:
                        stm = 1 if k.cur.split()[1] == "w" else -1
                        done_rows.append({"fen": k.cur, "cp": k.score * stm})
                    k.cur = None
                    k.score = None
        # watchdog
        now = time.time()
        for i, k in enumerate(kids):
            if k.cur and now - k.last_out > 90:
                pending.append(k.cur)
                k.kill()
                kids[i] = Child()


if __name__ == "__main__":
    main()
