#!/usr/bin/env python3
"""Labeling worker v3 — SHARD MODE: appends results to a per-node file that the
DB ingests via round-robin. No HTTP for results (only allocate), no write-lock
contention. File format: one JSON per line {fen, cp, depth}.
Usage: dbnet_worker3.py <node> <cores> <maxdepth> <api-url> <sf-binary> <shard-dir>"""
import json
import os
import select
import subprocess
import sys
import time
import urllib.request

NODE, CORES, MAXD, API, SF, SHARD = (sys.argv[1], int(sys.argv[2]),
                                     int(sys.argv[3]), sys.argv[4], sys.argv[5],
                                     sys.argv[6])
MINDEPTH = 1   # operator: store all evals from ply 1
BUF_TARGET = CORES * 6000
REPLENISH_S = 90


def api(path, body=None):
    for attempt in range(6):
        try:
            if body is None:
                with urllib.request.urlopen(f"{API}{path}", timeout=120) as r:
                    return json.load(r)
            req = urllib.request.Request(
                f"{API}{path}", data=json.dumps(body).encode(),
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
        self.p.stdin.write(
            b"uci\nsetoption name Hash value 8\nsetoption name Threads value 1\nisready\n")
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
    shard = open(f"{SHARD}/{NODE}.jsonl", "a", buffering=1)
    kids = [Child() for _ in range(CORES)]
    pending = []
    n_pos = 0
    last_replenish = time.time()
    while True:
        if time.time() - last_replenish > REPLENISH_S or len(pending) < CORES:
            want = min(BUF_TARGET - len(pending), 20000)
            if want > 0:
                r = api("/allocate", {"node": NODE, "engine": "sf17",
                                      "depth": MAXD, "n": want})
                pending.extend(r.get("fens", []))
                last_replenish = time.time()
            if not pending:
                time.sleep(5)
                continue
        for i, k in enumerate(kids):
            if not k.alive():
                kids[i] = Child()
            if k.cur is None and pending:
                k.cur = pending.pop(0)
                k.scores = {}
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
                ds = parse_depth_score(line)
                if ds:
                    d, v = ds
                    if d >= MINDEPTH:
                        k.scores[d] = v
                elif line.startswith("bestmove"):
                    if k.cur is not None and k.scores:
                        # stm-pov verbatim — the engine's output IS the label
                        shard.write(json.dumps(
                            {"fen": k.cur, "s": k.scores}) + "\n")
                        n_pos += 1
                    k.cur = None
                    k.scores = {}
        now = time.time()
        for i, k in enumerate(kids):
            if k.cur and now - k.last_out > 120:
                pending.append(k.cur)
                k.kill()
                kids[i] = Child()
        if n_pos and n_pos % 1000 == 0:
            print(f"{time.strftime('%H:%M:%S')} {NODE}: {n_pos:,} positions"
                  f" -> {SHARD}/{NODE}.jsonl", flush=True)


if __name__ == "__main__":
    main()
