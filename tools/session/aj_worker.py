#!/usr/bin/env python3
"""AJ-corpus d10 labeling worker. Reads a PGN shard, replays each kept
game, evaluates every position at depth 10 (ucinewgame, deterministic,
stm-pov cp verbatim), writes one JSONL line per game:
  {"h": movetext-sha1-12, "r": result, "we": whiteelo, "be": blackelo,
   "e": [cp per ply, d10], "fen0": first fen}
Usage: aj_worker.py <shard.pgn> <out.jsonl>"""
import hashlib
import json
import subprocess
import sys
import time

import chess
import chess.pgn

SF = "/srv/workspace/flychess/src/sf17/src/stockfish"

p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\nsetoption name Threads value 1\nisready\n")
p.stdin.flush()
while "readyok" not in p.stdout.readline():
    pass


def d10(fen):
    p.stdin.write(f"ucinewgame\nposition fen {fen}\ngo depth 10\n")
    p.stdin.flush()
    cp = None
    while True:
        line = p.stdout.readline()
        if not line:
            return None
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
        elif line.startswith("bestmove"):
            return cp


def main():
    shard, out_path = sys.argv[1], sys.argv[2]
    t0 = time.time()
    n = plies = 0
    with open(out_path, "w") as out:
        with open(shard, encoding="utf-8", errors="replace") as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                h = g.headers
                if "WhiteElo" not in h and "BlackElo" not in h:
                    continue
                board = g.board()
                fens = [board.fen()]
                try:
                    for mv in g.mainline_moves():
                        board.push(mv)
                        fens.append(board.fen())
                except Exception:
                    continue
                if len(fens) < 3:
                    continue
                evals = [d10(f) for f in fens]
                if any(e is None for e in evals):
                    continue
                sans = []
                b2 = g.board()
                for mv in g.mainline_moves():
                    sans.append(b2.san(mv))
                    b2.push(mv)
                out.write(json.dumps({
                    "h": hashlib.sha1(" ".join(sans).encode()).hexdigest()[:12],
                    "r": h.get("Result", "*"),
                    "we": h.get("WhiteElo", ""),
                    "be": h.get("BlackElo", ""),
                    "e": evals}) + "\n")
                n += 1
                plies += len(fens)
                if n % 50 == 0:
                    el = time.time() - t0
                    print(f"{time.strftime('%H:%M:%S')} {n} games "
                          f"{plies:,} plies ({plies/el:.1f}/s)", flush=True)
    print(f"SHARD DONE {n} games {plies:,} plies in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
