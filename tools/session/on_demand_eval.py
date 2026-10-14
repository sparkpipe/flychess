#!/usr/bin/env python3
"""On-demand eval service for the live page. Dead simple:
POST /eval {fen, depth} -> {cp}        — evaluates ONE position, returns instantly
Runs SF17 on the box, port 8124."""
import json
import os
import select
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SF17 = "/srv/workspace/flychess/src/sf17/src/stockfish"
PORT = 8124

p = subprocess.Popen([SF17], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\nsetoption name Threads value 1\nisready\n")
p.stdin.flush()
while "readyok" not in p.stdout.readline():
    pass


def eval_fen(fen, depth):
    p.stdin.write(f"position fen {fen}\ngo depth {depth}\n")
    p.stdin.flush()
    last = None
    while True:
        line = p.stdout.readline()
        if not line:
            return None
        if line.startswith("info") and " score " in line and " pv " in line:
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

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n))
        except Exception:
            return self._json(400, {"err": "bad json"})
        if self.path == "/eval":
            cp = eval_fen(body.get("fen", ""), int(body.get("depth", 12)))
            return self._json(200, {"cp": cp})
        self._json(404, {"err": "no"})


if __name__ == "__main__":
    print(f"on-demand eval on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
