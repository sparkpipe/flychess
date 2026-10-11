#!/usr/bin/env python3
"""evalfeed — the ONE process behind the UI (:8077). Operator design:

  the box serves the matches dir over plain HTTP  ->  the Mac PULLs the
  .pgn files  ->  eval sidecars live next to each PGN (name.pgn ->
  name.evals.json, keyed by fen)  ->  the Mac generates any missing
  sidecar entries (a d12 eval is ~70ms)  ->  serves games.json to the UI.

Sidecars ARE the persistence: a feed rebuild, a restart, a rewritten PGN
— evals re-derive from sidecars in milliseconds. All PULL, no box writes.
"""
import http.server
import json
import os
import re
import subprocess
import threading
import time
import urllib.request

import chess
import chess.pgn

HOME = os.path.expanduser("~/games-live")
PORT = 8077
SRC = "http://192.168.50.4:8078"          # box: plain static dir of PGNs
PGN_DIR = f"{HOME}/pgns"
SIDE_DIR = f"{HOME}/sidecars"
SF = "/Users/mac/sf-local/sf17/src/stockfish"
POOL = 4
D20_S_PER_PASS = 2.0
TAGS = {}
BOOK = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w": ("startpos", []),
    "rnbqkbnr/ppp1pppp/8/3p4/3PP3/8/PPP2PPP/RNBQKBNR b":
        ("BDG (1.d4 d5 2.e4)", ["d2d4", "d7d5", "e2e4"]),
    "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b":
        ("French Advance (3.e5)", ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5"]),
}

for d in (PGN_DIR, SIDE_DIR):
    os.makedirs(d, exist_ok=True)

# ---- SF pool: one search -> (cp white-pov, pv uci) ----
pool = []
pool_mu = threading.Lock()


def spawn():
    p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\nsetoption name Threads value 1\nisready\n")
    p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    return p


for _ in range(POOL):
    pool.append(spawn())


def search(fen, depth):
    with pool_mu:
        p = pool.pop(0)
    try:
        p.stdin.write(f"ucinewgame\nposition fen {fen}\ngo depth {depth}\n")
        p.stdin.flush()
        cp, pv = None, ""
        while True:
            line = p.stdout.readline()
            if not line:
                raise EOFError
            if line.startswith("info") and not line.startswith("info string") \
                    and " score " in line:
                t = line.split()
                for i, x in enumerate(t):
                    if i and t[i-1] == "score" and x in ("cp", "mate"):
                        try:
                            v = int(t[i+1])
                        except (ValueError, IndexError):
                            break
                        raw = ((10000 + min(abs(v), 900)) * (1 if v > 0 else -1)
                               if x == "mate" else v)
                        cp = raw * (1 if fen.split()[1] == "w" else -1)
                        break
                if " pv " in line:
                    pv = line.split(" pv ", 1)[1].strip()
            elif line.startswith("bestmove"):
                return cp, pv
    except Exception:
        try:
            p.kill()
        except Exception:
            pass
        p = spawn()
        return None, ""
    finally:
        pool.append(p)


# ---- PGN pulling ----
def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def list_pgns():
    html = fetch(SRC + "/").decode()
    return sorted(set(re.findall(r'href="([^"?/]*\.pgn)"', html)))


def pull_pgns():
    changed = []
    for name in list_pgns():
        want = f"{PGN_DIR}/{name}"
        have = os.path.getsize(want) if os.path.exists(want) else -1
        data = fetch(f"{SRC}/{name}")
        if len(data) != have:
            with open(want + ".tmp", "wb") as f:
                f.write(data)
            os.replace(want + ".tmp", want)
            changed.append(name)
    return changed


# ---- PGN -> games (same shape the UI already consumes) ----
def book_san_for(fen0):
    start = " ".join(fen0.split()[:2])
    label, ucis = BOOK.get(start, ("book position", []))
    if not ucis:
        return label, []
    b = chess.Board()
    sans = []
    try:
        for u in ucis:
            mv = chess.Move.from_uci(u)
            sans.append(b.san(mv))
            b.push(mv)
    except Exception:
        return label, []
    return label, sans


def parse_pgn(path):
    games = []
    with open(path) as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            board = g.board()
            sans, ucis, fens = [], [], [board.fen()]
            for mv in g.mainline_moves():
                sans.append(board.san(mv))
                ucis.append(mv.uci())
                board.push(mv)
                fens.append(board.fen())
            h = g.headers
            label, book_san = book_san_for(fens[0])
            games.append({
                "white": h.get("White", "?"), "black": h.get("Black", "?"),
                "result": h.get("Result", "*"), "ply": len(sans),
                "termination": h.get("Termination", "normal"),
                "open": h.get("Result", "*") == "*",
                "opening": label, "book_san": book_san,
                "white_elo": "", "black_elo": "",
                "date": h.get("Date", ""), "time": h.get("GameEndTime", ""),
                "sans": sans, "ucis": ucis, "fens": fens,
            })
    return games


# ---- sidecars: name.evals.json = {fen: {"12": cp, "12pv": "cp|mv ...", ...}} ----
def sidecar_path(pgn_name):
    return f"{SIDE_DIR}/{pgn_name[:-4]}.evals.json"


def load_json(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def atomic_json(obj, path):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def fill_sidecars(games, side, deadline_d20):
    for g in games:
        for depth in (12, 20):
            if depth == 20 and time.time() > deadline_d20:
                break
            for fen in g["fens"]:
                e = side.get(fen)
                if e and str(depth) in e and f"{depth}pv" in e:
                    continue
                cp, pv = search(fen, depth)
                if cp is None:
                    continue
                e = side.setdefault(fen, {})
                e[str(depth)] = cp
                e[f"{depth}pv"] = f"{cp}|{pv}" if pv else "|"
    return side


def build(pgn_names):
    doc = []
    for name in pgn_names:
        games = parse_pgn(f"{PGN_DIR}/{name}")
        if not games:
            continue
        side = load_json(sidecar_path(name), {})
        for g in games:
            for depth in (12, 20):
                g[f"evals_d{depth}"] = [side.get(f, {}).get(str(depth))
                                        for f in g["fens"]]
                g[f"pv_d{depth}"] = [side.get(f, {}).get(f"{depth}pv") or None
                                     for f in g["fens"]]
        doc.append({"tag": TAGS.get(name, name[:-4]), "games": games})
    return doc


def loop():
    while True:
        try:
            pull_pgns()
            names = sorted(n for n in os.listdir(PGN_DIR) if n.endswith(".pgn"))
            doc = build(names)
            deadline = time.time() + D20_S_PER_PASS
            for i, m in enumerate(doc):
                side = load_json(sidecar_path(names[i]), {})
                n0 = len(side)
                fill_sidecars(m["games"], side, deadline)
                if len(side) != n0:
                    atomic_json(side, sidecar_path(names[i]))
                for g in m["games"]:
                    for depth in (12, 20):
                        g[f"evals_d{depth}"] = [
                            side.get(f, {}).get(str(depth)) for f in g["fens"]]
                        g[f"pv_d{depth}"] = [
                            side.get(f, {}).get(f"{depth}pv") or None
                            for f in g["fens"]]
            atomic_json(doc, f"{HOME}/games.json")
        except Exception as e:
            with open(f"{HOME}/feed_errors.log", "a") as el:
                el.write(f"{time.strftime('%H:%M:%S')} {e}\n")
        time.sleep(3)


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HOME, **kw)

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path == "/focus":
            n = int(self.headers.get("Content-Length", 0))
            try:
                open(f"{HOME}/focus.json", "wb").write(self.rfile.read(n))
                self.send_response(204)
            except Exception:
                self.send_response(500)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()


class S(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    with S(("127.0.0.1", PORT), H) as s:
        print(f"evalfeed on http://localhost:{PORT}/ (pulling {SRC})")
        s.serve_forever()
