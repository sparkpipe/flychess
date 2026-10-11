#!/usr/bin/env python3
"""Crash battery for a local SF build: quick-mate/low-material positions that
segfaulted the LTO apple-silicon build, plus healthy controls."""
import subprocess
import sys
import time

SF = sys.argv[1] if len(sys.argv) > 1 else "/Users/mac/sf-local/sf17/src/stockfish"

CASES = [
    ("3men-QKvK   ", "k7/8/1K6/8/8/8/8/7Q w - - 0 1"),
    ("QKvK-4men   ", "7k/8/6K1/8/8/8/8/7Q w - - 0 1"),
    ("mate-in-1   ", "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"),
    ("middlegame  ", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("rook-endgame", "8/8/4kpp1/3p4/p1pP4/P1P2PP1/4K3/6R1 w - - 0 1"),
    ("3men-QKvK-2 ", "k7/8/1K6/8/8/8/8/7Q w - - 0 1"),
]


def run_one(label, fen, depth=12):
    p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\nsetoption name Threads value 1\nisready\n")
    p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    p.stdin.write(f"ucinewgame\nposition fen {fen}\ngo depth {depth}\n")
    p.stdin.flush()
    ok, last = False, ""
    t0 = time.time()
    while time.time() - t0 < 20:
        line = p.stdout.readline()
        if not line:
            break
        last = line.strip()
        if line.startswith("bestmove"):
            ok = True
            break
    print(f"{label}: survived={ok} rc={p.poll()} last={last[:60]}")
    try:
        p.stdin.write("quit\n")
        p.stdin.flush()
    except Exception:
        pass
    time.sleep(0.2)
    if p.poll() is None:
        p.kill()


for label, fen in CASES:
    run_one(label, fen)
