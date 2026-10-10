#!/usr/bin/env python3
"""Pilot verdict: wait-for-bestmove eval scores with pilot net in the piece_down slot."""
import subprocess
import sys

N16 = "/srv/workspace/flychess/src/Stockfish/src/stockfish"
E16 = "/srv/workspace/chess-active/engine16"
pilot = sys.argv[1]

MAP = ["tb", "mvr", "rv2m", "qvmat", "nvb", None, "oppb", "dv_Q",
       "dv_R", "dv_rest", "op_pawnimb", "op_even_l0", "op_even_l1",
       "op_even_l2p", "mg_unsafe", "mg_safe"]
TESTS = [
    ("T2_wUpQ_w  expect ~+900", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("T3_wUpQ_b  expect ~-900", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"),
    ("T4_wDownR_w expect ~-500", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w kq - 0 1"),
    ("T5_wDownR_b expect ~+500", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR b kq - 0 1"),
]


class Eng:
    def __init__(self):
        self.p = subprocess.Popen([N16], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("uci")
        self.wait("uciok")
        for i, name in enumerate(MAP):
            path = pilot if name is None else f"{E16}/{name}.nnue"
            opt = "EvalFile" if i == 0 else f"EvalFile{i+1}"
            self.send(f"setoption name {opt} value {path}")
        self.send("isready")
        self.wait("readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n")

    def wait(self, tok):
        while True:
            l = self.p.stdout.readline()
            if not l or tok in l:
                return

    def score(self, fen, depth=8):
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        last = ""
        while True:
            l = self.p.stdout.readline()
            if not l:
                return "EOF"
            if l.startswith("info ") and " score " in l and " pv " in l:
                t = l.split()
                for i, x in enumerate(t):
                    if x == "score":
                        last = t[i + 1] + " " + t[i + 2]
            if l.startswith("bestmove"):
                return last


e = Eng()
print(f"pilot: {pilot}")
for name, fen in TESTS:
    print(f"  {name}: {e.score(fen)}")
e.send("quit")
