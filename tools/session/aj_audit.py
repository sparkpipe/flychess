#!/usr/bin/env python3
"""End-to-end audit of the AJ d10 pipeline: take the first games of a shard
currently being processed, replay + re-eval them with a fresh SF17 (same
deterministic protocol), and compare against the worker's output arrays."""
import glob
import hashlib
import json
import subprocess
import sys

import chess
import chess.pgn

SF = "/srv/workspace/flychess/src/sf17/src/stockfish"
OUT = "/srv/workspace/chess-active/aj-kept/out"

claims = sorted(glob.glob("/srv/workspace/chess-active/aj-kept/aj.*.claim*"))
if not claims:
    sys.exit("no shard currently claimed")
shard = claims[0]
base = shard.split("/")[-1].split(".claim")[0]
if base.endswith(".pgn"):
    base = base[:-4]
out_file = f"{OUT}/{base}.jsonl"
done = {}
for line in open(out_file):
    r = json.loads(line)
    done[r["h"]] = r["e"]

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


checked = mismatches = missing = 0
with open(shard, encoding="utf-8", errors="replace") as f:
    while checked < 5:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        if "WhiteElo" not in g.headers and "BlackElo" not in g.headers:
            continue
        board = g.board()
        fens = [board.fen()]
        try:
            for mv in g.mainline_moves():
                board.push(mv)
                fens.append(board.fen())
        except Exception:
            continue
        if len(fens) < 10:
            continue
        sans = []
        b2 = g.board()
        for mv in g.mainline_moves():
            sans.append(b2.san(mv))
            b2.push(mv)
        h = hashlib.sha1(" ".join(sans).encode()).hexdigest()[:12]
        if h not in done:
            missing += 1
            continue
        truth = [d10(x) for x in fens]
        stored = done[h]
        checked += 1
        if truth != stored:
            mismatches += 1
            diff = [(i, a, b) for i, (a, b) in enumerate(zip(truth, stored))
                    if a != b][:3]
            print(f"MISMATCH {h}: first diffs (ply, truth, stored): {diff}")
        else:
            print(f"MATCH {h}: all {len(truth)} plies identical")
print(f"\nAUDIT: {checked} games compared, {mismatches} mismatches, "
      f"{missing} not-yet-written")
