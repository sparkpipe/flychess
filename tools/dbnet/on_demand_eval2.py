#!/usr/bin/env python3
"""On-demand eval service. Port 8124 (localhost). CORS enabled for the UI.

Pool of SF17 processes behind a free-list queue: several UI clients can
sweep concurrently at full speed. Each eval runs ucinewgame -> position ->
go depth N and reads to its own bestmove on a process it owns exclusively,
so UCI streams can never interleave. A dead process is respawned on error.
"""
import json
import queue
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SF17 = "/Users/mac/sf-local/sf17/src/stockfish"
POOL = 4

pool = queue.Queue()


def _spawn():
    p = subprocess.Popen([SF17], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\nsetoption name Threads value 1\nisready\n")
    p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    return p


for _ in range(POOL):
    pool.put(_spawn())


def _search(p, fen, depth):
    p.stdin.write(f"ucinewgame\nposition fen {fen}\ngo depth {depth}\n")
    p.stdin.flush()
    last = None
    while True:
        line = p.stdout.readline()
        if not line:
            raise EOFError("sf died")
        if line.startswith("info") and not line.startswith("info string") \
                and " score " in line:
            t = line.split()
            for i, x in enumerate(t):
                if i > 0 and t[i-1] == "score" and x in ("cp", "mate"):
                    try:
                        v = int(t[i+1])
                    except (ValueError, IndexError):
                        continue
                    last = ((10000 + min(abs(v), 900)) * (1 if v > 0 else -1)
                            if x == "mate" else v)
        elif line.startswith("bestmove"):
            stm = 1 if fen.split()[1] == "w" else -1
            return last * stm if last is not None else None


def eval_fen(fen, depth):
    depth = max(1, min(int(depth), 30))
    p = pool.get()
    try:
        try:
            cp = _search(p, fen, depth)
        except (EOFError, BrokenPipeError, OSError):
            try:
                p.kill()
            except Exception:
                pass
            p = _spawn()          # replace the dead one and retry once
            cp = _search(p, fen, depth)
        return cp
    except Exception:
        return None
    finally:
        pool.put(p)


class H(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n))
        except Exception:
            return self._json(400, {"err": "bad json"})
        if self.path == "/eval":
            try:
                cp = eval_fen(body.get("fen", ""), body.get("depth", 12))
            except Exception as e:
                return self._json(500, {"err": str(e)})
            return self._json(200, {"cp": cp})
        self._json(404, {"err": "no"})


if __name__ == "__main__":
    print(f"on-demand eval pool({POOL}) on 127.0.0.1:8124", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8124), H).serve_forever()
