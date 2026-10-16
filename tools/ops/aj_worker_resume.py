#!/usr/bin/env python3
"""aj_worker_resume.py — aj_worker.py with shard resume.

Identical eval + output format to aj_worker.py. Difference: if the output
JSONL already exists, count its lines N, skip the first N games that pass
the eval-free filters (Elo headers, parseable mainline, >=3 plies), and
APPEND. With a healthy engine pipe, kept-order == filter-order exactly
(eval-None drops only occur on engine death, which invalidates the whole
run anyway); a last-line hash guard catches a one-game misalignment from
any pathological case by skipping the duplicate.

Usage: aj_worker_resume.py <shard.pgn> <out.jsonl>"""
import hashlib
import json
import os
import subprocess
import sys
import time

import chess
import chess.pgn

import os
SF = os.environ.get("SF17", "/mnt/flychess/flychess/src/sf17/src/stockfish")

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
    skip = 0
    last_h = None
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    skip += 1
                    try:
                        last_h = json.loads(line).get("h")
                    except json.JSONDecodeError:
                        pass
        print(f"RESUME: {skip} games already in {out_path}, skipping + appending",
              flush=True)
    n = plies = 0
    skipped = 0
    with open(out_path, "a") as out:
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
                # eval-free filters passed: this game occupies a kept-slot
                if skipped < skip:
                    skipped += 1
                    continue
                evals = [d10(f) for f in fens]
                if any(e is None for e in evals):
                    continue
                sans = []
                b2 = g.board()
                for mv in g.mainline_moves():
                    sans.append(b2.san(mv))
                    b2.push(mv)
                gh = hashlib.sha1(" ".join(sans).encode()).hexdigest()[:12]
                if skip and gh == last_h and n == 0:
                    print(f"RESUME-GUARD: first new game hash matches last "
                          f"file line ({gh}) — skipping duplicate", flush=True)
                    last_h = None
                    continue
                out.write(json.dumps({
                    "h": gh,
                    "r": h.get("Result", "*"),
                    "we": h.get("WhiteElo", ""),
                    "be": h.get("BlackElo", ""),
                    "e": evals}) + "\n")
                out.flush()
                n += 1
                plies += len(fens)
                if n % 50 == 0:
                    el = time.time() - t0
                    print(f"{time.strftime('%H:%M:%S')} {n} games "
                          f"{plies:,} plies ({plies/el:.1f}/s)", flush=True)
    print(f"SHARD DONE +{n} games {plies:,} plies in {time.time()-t0:.0f}s "
          f"(skipped {skipped})", flush=True)


if __name__ == "__main__":
    main()
