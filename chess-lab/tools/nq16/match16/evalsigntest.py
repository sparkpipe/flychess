#!/usr/bin/env python3
"""Eval sign test: known-truth positions through nQ13 and n16 (match wiring).
Reads until bestmove per position; prints stm-pov scores side by side."""
import subprocess
import sys

NQ = "/mnt/cold-raid6/chess-audit/nq13_binary_backup"
NQN = "/mnt/cold-raid6/chess-audit/nets"
N16 = "/srv/workspace/flychess/src/Stockfish/src/stockfish"
E16 = "/srv/workspace/chess-active/engine16"

NQ_MAP = ["balanced_l0", "balanced_l1", "balanced_l2", "balanced_l3", "nvb",
          "nvr", "bvr", "rv2m", "qvmat", "oppb", "dvoretsky", "exchanges", "tactics"]
N16_MAP = ["tb", "mvr", "rv2m", "qvmat", "nvb", "piece_down", "oppb", "dv_Q",
           "dv_R", "dv_rest", "op_pawnimb", "op_even_l0", "op_even_l1",
           "op_even_l2p", "mg_unsafe", "mg_safe"]

# (name, fen, expected stm-pov sign, note)
TESTS = [
    ("T1_startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "~0", ""),
    ("T2_wUpQ_w", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "BIG+", "white up Q, w to move"),
    ("T3_wUpQ_b", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "BIG-", "white up Q, b to move"),
    ("T4_wDownR_w", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w kq - 0 1", "BIG-", "white down R, w to move"),
    ("T5_wDownR_b", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR b kq - 0 1", "BIG+", "white down R, b to move"),
    ("T6_KRK_w", "8/8/8/8/8/4k3/8/4K2R w K - 0 1", "BIG+", "KRK white winning"),
    ("T7_qvmat", "4k3/8/8/8/8/2n2r2/8/2Q1K3 w - - 0 1", "BIG+", "Q vs R+N, white up, w to move"),
    ("T8_qvmat_b", "4k3/8/8/8/8/2n2r2/8/2Q1K3 b - - 0 1", "BIG-", "same, b to move"),
]


class Eng:
    def __init__(self, path, slotmap, netdir):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("uci")
        self.wait("uciok")
        for i, name in enumerate(slotmap):
            opt = "EvalFile" if i == 0 else f"EvalFile{i+1}"
            self.send(f"setoption name {opt} value {netdir}/{name}.nnue")
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
            if l.startswith("info ") and "score " in l and " pv " in l:
                for i, t in enumerate(l.split()):
                    if t == "score":
                        last = l.split()[i + 1] + " " + l.split()[i + 2]
                        break
            if l.startswith("bestmove"):
                return last or "none"


e1 = Eng(NQ, NQ_MAP, NQN)
e2 = Eng(N16, N16_MAP, E16)
print(f"{'test':<14} {'expect':<7} {'nQ13':<14} {'n16':<14}  note")
for name, fen, exp, note in TESTS:
    s1 = e1.score(fen)
    s2 = e2.score(fen)
    print(f"{name:<14} {exp:<7} {s1:<14} {s2:<14}  {note}")
e1.send("quit")
e2.send("quit")
