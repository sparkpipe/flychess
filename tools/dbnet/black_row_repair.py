#!/usr/bin/env python3
"""REPAIR: black-stm miniature labels carry a white-pov flip from the worker bug.
Re-evaluate every stm='b' sf17-d12 label against fresh SF17 and overwrite with the
ground-truth stm-pov value. White-stm rows are correct under both eras — untouched.
Uses the dbnet worker's proven raw-FD multiplexing; direct DB updates."""
import os
import select
import sqlite3
import subprocess
import sys
import time

SF17 = "/srv/workspace/flychess/src/sf17/src/stockfish"
DB = "/srv/workspace/chess-active/store/games.db"
NW = 12


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
    con = sqlite3.connect(DB, check_same_thread=False)
    con.execute("pragma journal_mode=wal")
    fens = [r[0] for r in con.execute(
        "select l.fen from labels l where l.engine='sf17' and l.depth=12"
        " and substr(l.fen, instr(l.fen,' ')+1, 1) = 'b'"
        " and by_node = 'box-relabel17'")]
    print(f"black-stm miniature labels to re-verify: {len(fens):,}", flush=True)
    kids = [Child() for _ in range(NW)]
    pending = list(fens)
    fixes = checked = 0
    out_rows = []
    t0 = time.time()
    qi = 0
    while qi < len(pending) or any(k.cur for k in kids):
        for i, k in enumerate(kids):
            if not k.alive():
                kids[i] = Child()
            if k.cur is None and qi < len(pending):
                k.cur = pending[qi]
                qi += 1
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
                    qi -= 1
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
                        out_rows.append((k.score, k.cur))  # stm-pov verbatim
                        checked += 1
                    k.cur = None
                    k.score = None
        if len(out_rows) >= 50000:
            con.executemany(
                "update labels set cp=? where fen=? and engine='sf17' and depth=12",
                out_rows)
            con.commit()
            out_rows = []
            rate = checked / max(time.time() - t0, 1) * 60
            print(f"{checked:,} checked ({rate:,.0f}/min)", flush=True)
    if out_rows:
        con.executemany(
            "update labels set cp=? where fen=? and engine='sf17' and depth=12",
            out_rows)
        con.commit()
    print(f"REPAIR DONE: {checked:,} black-stm labels re-verified against SF17",
          flush=True)


if __name__ == "__main__":
    main()
