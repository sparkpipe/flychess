#!/usr/bin/env python3
"""SF17 miniature relabel v3 — stall-proof, raw-FD IO (select-safe).
16 SF17 children on non-blocking pipes; hung child killed+respawned, work re-queued.
Output: append-only miniature_labels17.tsv.part (resume-safe). Mate = 10000+N."""
import os
import select
import subprocess
import time

SF17 = "/srv/workspace/flychess/src/sf17/src/stockfish"
SRC = "/srv/workspace/chess-active/miniature_rows.tsv"
OUT = "/srv/workspace/chess-active/miniature_labels17.tsv.part"
NW = 16
STALL_S = 90


class Child:
    def __init__(self, idx):
        self.idx = idx
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
        self.p.stdin.write(f"position fen {fen}\ngo depth 12\n".encode())
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
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            done.add(line.split("\t")[0])
    print(f"resume: {len(done):,}", flush=True)
    todo = [l.split("\t")[0] for l in open(SRC) if l.split("\t")[0] not in done]
    print(f"todo: {len(todo):,}", flush=True)
    out = open(OUT, "a")
    kids = [Child(i) for i in range(NW)]
    pending = [[] for _ in range(NW)]
    qi = 0

    def feed(i):
        nonlocal qi
        while len(pending[i]) < 8 and qi < len(todo):
            pending[i].append(todo[qi])
            qi += 1

    for i in range(NW):
        feed(i)
    n = len(done)
    t0 = time.time()
    idle_rounds = 0
    while True:
        work_left = qi < len(todo) or any(pending[i] or kids[i].cur for i in range(NW))
        if not work_left:
            break
        # dispatch
        for i, k in enumerate(kids):
            if not k.alive():
                if k.cur:
                    pending[i].append(k.cur)
                kids[i] = Child(i)
            if k.cur is None and pending[i]:
                k.cur = pending[i].pop(0)
                k.score = None
                k.send(k.cur)
        fds = {}
        for i, k in enumerate(kids):
            if k.alive() and (k.cur or pending[i]):
                fds[k.fd()] = i
        if not fds:
            time.sleep(0.2)
            continue
        try:
            r, _, _ = select.select(list(fds), [], [], 1.0)
        except (OSError, ValueError):
            continue
        got = False
        for fd in r:
            i = fds[fd]
            k = kids[i]
            try:
                data = os.read(fd, 65536)
            except BlockingIOError:
                continue
            if not data:
                if k.cur:
                    pending[i].append(k.cur)
                    k.cur = None
                k.kill()
                kids[i] = Child(i)
                continue
            got = True
            k.buf += data
            while b"\n" in k.buf:
                raw, k.buf = k.buf.split(b"\n", 1)
                line = raw.decode(errors="replace")
                k.last_out = time.time()
                if line.startswith("info") and " score " in line:
                    s = parse_score(line)
                    if s is not None:
                        k.score = s  # last score line wins (deepest iteration)
                elif line.startswith("bestmove"):
                    if k.cur is not None:
                        if k.score is not None:
                            stm = 1 if k.cur.split()[1] == "w" else -1
                            out.write(f"{k.cur}\t{k.score * stm}\n")
                            n += 1
                            if n % 50000 < NW:
                                out.flush()
                                rate = n and (n - len(done) + len(done)) // max(1, (time.time() - t0) / 60)
                                print(f"{n:,} total ({rate:,.0f}/min)", flush=True)
                        k.cur = None
                        k.score = None
                    feed(i)
        if not got and not r:
            idle_rounds += 1
        else:
            idle_rounds = 0
        # stall watchdog: child with assigned work but no output for STALL_S
        now = time.time()
        for i, k in enumerate(kids):
            if k.cur and now - k.last_out > STALL_S:
                print(f"child {i} stalled — respawn", flush=True)
                if k.cur:
                    pending[i].append(k.cur)
                k.kill()
                kids[i] = Child(i)
    out.flush()
    print(f"COMPLETE {n:,}", flush=True)


if __name__ == "__main__":
    main()
