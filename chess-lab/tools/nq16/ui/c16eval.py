#!/usr/bin/env python3
"""correct16 top-3 eval service (box, port 8127). The n16 fork binary with
the round-2 corrected nets, MultiPV 3. Same pattern as g1eval."""
import json
import subprocess
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SF = "/srv/workspace/flychess/src/Stockfish/src/stockfish"
NETS = "/srv/workspace/chess-active/engine16-correct"

EXPERTS = ["tb", "mvr", "rv2m", "qvmat", "nvb", "piece_down", "oppb", "dv_Q",
           "dv_R", "dv_rest", "op_pawnimb", "op_even_l0", "op_even_l1",
           "op_even_l2p", "mg_unsafe", "mg_safe"]

p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
opts = "uci\n"
for i, e in enumerate(EXPERTS):
    slot = "" if i == 0 else str(i + 1)
    opts += f"setoption name EvalFile{slot} value {NETS}/{e}.nnue\n"
opts += "setoption name MultiPV value 3\nisready\n"
p.stdin.write(opts)
p.stdin.flush()
while "readyok" not in p.stdout.readline():
    pass

LOCK = threading.Lock()


def top(fen, depth):
    with LOCK:
        p.stdin.write(f"ucinewgame\nposition fen {fen}\ngo depth {depth}\n")
        p.stdin.flush()
        lines = {}
        while True:
            ln = p.stdout.readline()
            if not ln:
                return []
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
    print("correct16 top-3 on 0.0.0.0:8127", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8127), H).serve_forever()
