#!/usr/bin/env python3
"""Interactive engine test — send commands one at a time, wait for responses,
mimicking how fastchess drives the engine. Compare single-PV vs MultiPV."""
import subprocess
import time
import sys

SF = "/srv/workspace/flychess/src/Stockfish/src/stockfish"
NET = "/srv/workspace/chess-active/runs-general3b/best.nnue"
FEN = "r1bq1rk1/2p1bppp/p1n2n2/1p1pp3/4P3/P1P1BN2/1P1PBPPP/RN1Q1RK1 w - - 0 11"

def run_test(multipv, depth):
    p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    # interactive handshake
    p.stdin.write("uci\n"); p.stdin.flush()
    while "uciok" not in p.stdout.readline(): pass
    p.stdin.write(f"setoption name EvalFile value {NET}\n"); p.stdin.flush()
    if multipv > 1:
        p.stdin.write(f"setoption name MultiPV value {multipv}\n"); p.stdin.flush()
    p.stdin.write("setoption name Threads value 1\nisready\n"); p.stdin.flush()
    while "readyok" not in p.stdout.readline(): pass

    # set position, then search — with a pause between commands
    p.stdin.write(f"position fen {FEN}\n"); p.stdin.flush()
    time.sleep(0.1)
    p.stdin.write(f"go depth {depth}\n"); p.stdin.flush()

    best = None
    depth_reached = 0
    nodes = 0
    last_pv = ""
    t0 = time.time()
    while time.time() - t0 < 60:
        line = p.stdout.readline()
        if not line: break
        if line.startswith("info") and " depth " in line and " pv " in line:
            t = line.split()
            depth_reached = int(t[t.index("depth") + 1])
            try: nodes = int(t[t.index("nodes") + 1])
            except: pass
            pv_start = line.index(" pv ") + 4
            last_pv = line[pv_start:].strip()
        elif line.startswith("bestmove"):
            best = line.split()[1]
            break

    elapsed = time.time() - t0
    p.stdin.write("quit\n"); p.stdin.flush(); p.wait()
    return best, depth_reached, nodes, last_pv, elapsed

print("=== interactive single-PV d20 (match mode) ===")
best, dep, nodes, pv, el = run_test(1, 20)
print(f"bestmove: {best}  depth: {dep}  nodes: {nodes:,}  time: {el:.1f}s")
print(f"pv: {pv[:60]}")

print("\n=== interactive MultiPV-3 d20 (analysis mode) ===")
best, dep, nodes, pv, el = run_test(3, 20)
print(f"bestmove: {best}  depth: {dep}  nodes: {nodes:,}  time: {el:.1f}s")
print(f"pv: {pv[:60]}")
