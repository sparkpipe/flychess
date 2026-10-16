#!/usr/bin/env python3
"""general3B top-3 eval service (box, port 8127). Single net, MultiPV 3.
Same pattern as g1eval (serialized with respawn)."""
import json
import os
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHESS_DATA = os.environ.get("CHESS_DATA") or sys.exit(
    "CHESS_DATA not set (data root holding runs-*/best.nnue)")
SF = os.environ.get("FORK_BIN") or sys.exit(
    "FORK_BIN not set (n16-fork stockfish binary)")
NET = os.environ.get("NET") or f"{CHESS_DATA}/runs-general3b/best.nnue"
PORT = int(os.environ.get("PORT", "8127"))

LOCK = threading.Lock()


def spawn():
    p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write(f"uci\nsetoption name EvalFile value {NET}\n"
                  "setoption name MultiPV value 3\nisready\n")
    p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    return p


p = spawn()


def top(fen, depth):
    with LOCK:
        try:
            p.stdin.write(f"ucinewgame\nposition fen {fen}\ngo depth {depth}\n")
            p.stdin.flush()
            lines = {}
            while True:
                ln = p.stdout.readline()
                if not ln:
                    raise EOFError
                if ln.startswith("info") and " multipv " in ln \
                        and " score " in ln and " pv " in ln:
                    t = ln.split()
                    try:
                        mi = int(t[t.index("multipv") + 1])
                        kind = t[t.index("score") + 1]
                        v = int(t[t.index("score") + 2])
                    except (ValueError, IndexError):
                        continue
                    raw = ((10000 + min(abs(v), 900)) * (1 if v > 0 else -1)
                           if kind == "mate" else v)
                    lines[mi] = (raw * (1 if fen.split()[1] == "w" else -1),
                                 ln.split(" pv ", 1)[1].strip())
                elif ln.startswith("bestmove"):
                    return [lines[k] for k in sorted(lines)][:3]
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
            return []


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/top?"):
            q = urllib.parse.parse_qs(self.path.split("?", 1)[1])
            fen = urllib.parse.unquote(q.get("fen", [""])[0])
            depth = max(1, min(int(q.get("depth", ["12"])[0]), 30))
            try:
                out = top(fen, depth)
            except Exception:
                out = []
            b = json.dumps({"lines": out}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        self.send_response(404)
        self.end_headers()


if __name__ == "__main__":
    print(f"general3B top-3 on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
