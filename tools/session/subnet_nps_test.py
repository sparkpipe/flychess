#!/usr/bin/env python3
"""Sub-net NPS test: ONE engine slot (piece_down) with TWO nets loaded —
piece_up net and piece_down net — the engine picks by material sign.
Measures nps with (a) single mixed net, (b) two sub-nets in the slot.
If nps(a) ~= nps(b), the bimodal split costs nothing at search time."""
import subprocess

N16 = "/srv/workspace/flychess/src/Stockfish/src/stockfish"
E = "/srv/workspace/chess-active/engine16-baseline"
MAP = ["tb", "mvr", "rv2m", "qvmat", "nvb", "piece_down", "oppb", "dv_Q",
       "dv_R", "dv_rest", "op_pawnimb", "op_even_l0", "op_even_l1",
       "op_even_l2p", "mg_unsafe", "mg_safe"]
FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]


def bench(extra_files):
    p = subprocess.Popen([N16], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, bufsize=1)
    p.stdin.write("uci\n")
    p.stdin.flush()
    while "uciok" not in p.stdout.readline():
        pass
    for i, n in enumerate(MAP):
        opt = "EvalFile" if i == 0 else f"EvalFile{i+1}"
        p.stdin.write(f"setoption name {opt} value {E}/{n}.nnue\n")
    p.stdin.write("isready\n")
    p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    total_n = total_ms = 0
    for fen in FENS:
        p.stdin.write(f"position fen {fen}\ngo nodes 300000\n")
        p.stdin.flush()
        n = s = 0
        while True:
            l = p.stdout.readline()
            if not l:
                break
            if l.startswith("info ") and " nodes " in l and " nps " in l:
                t = l.split()
                n = int(t[t.index("nodes") + 1])
                s = int(t[t.index("nps") + 1])
            elif l.startswith("bestmove"):
                if s > 0:
                    total_n += n
                    total_ms += n * 1000.0 / s
                break
    p.stdin.write("quit\n")
    return total_n / (total_ms / 1000) if total_ms else 0


a = bench([])
print(f"single mixed net (piece_down): {a:,.0f} nps")
b = bench([])
print(f"repeat:                        {b:,.0f} nps")
# NOTE: two-net slot test requires the engine sub-net code (router picks by
# material sign within one slot) — that code is next. This baseline confirms
# the mixed-net reference number for comparison when it lands.
