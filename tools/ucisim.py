"""Raw-UCI match driver (no python-chess engine wrapper - avoids its fragility).
Plays N games our-engine vs a reference engine at fixed movetime.
Usage: ucisim.py <our_bin> <opp_bin> <movetime_ms> <games> [setoption lines...]
Env: PHASE_MOE etc pass through. Prints per-game results + crash detection.
"""
import sys
import subprocess
import random
import time

our_bin, opp_bin = sys.argv[1], sys.argv[2]
movetime = int(sys.argv[3])
games = int(sys.argv[4])
extra_opts = sys.argv[5:]


class UCI:
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True,
                                  bufsize=1)
        self.cmd("uci")
        self.wait("uciok")

    def cmd(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def wait(self, token):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.startswith(token):
                return line

    def setoption(self, name, value):
        self.cmd("setoption name %s value %s" % (name, value))

    def go(self, movetime, pos_line):
        self.cmd(pos_line)
        self.cmd("go movetime %d" % movetime)
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died during search")
            if line.startswith("bestmove"):
                return line.split()[1]

    def quit(self):
        try:
            self.cmd("quit")
            self.p.wait(timeout=3)
        except Exception:
            self.p.kill()


def play(our, opp, our_white, seed):
    import chess  # only for game logic, not engines
    board = chess.Board()
    rng = random.Random(seed)
    opening = []
    for _ in range(rng.randrange(4, 8)):
        if board.is_game_over():
            break
        m = rng.choice(list(board.legal_moves))
        opening.append(m.uci())
        board.push(m)
    moves = []
    for _ in range(220):
        eng = our if (board.turn == chess.WHITE) == our_white else opp
        pos_line = "position startpos moves %s" % " ".join(opening + moves)
        mv = eng.go(movetime, pos_line)
        try:
            m = chess.Move.from_uci(mv)
        except Exception:
            break
        if m not in board.legal_moves:
            break
        board.push(m)
        moves.append(mv)
        if board.is_game_over():
            break
    res = board.result()
    if res in ("1/2-1/2", "*"):
        return 0.5, len(moves)
    return (1.0 if (res == "1-0") == our_white else 0.0), len(moves)


our = UCI(our_bin)
for opt in extra_opts:
            # opt format: "Name=Value"
    our.setoption(*[x.strip() for x in opt.split("=", 1)])
our.setoption("Threads", "1")
our.setoption("Hash", "64")
our.cmd("isready"); our.wait("readyok")
opp = UCI(opp_bin)
opp.setoption("Skill Level", "15")
opp.setoption("Threads", "1")
opp.setoption("Hash", "64")
opp.cmd("isready"); opp.wait("readyok")

scores = []
for i in range(games):
    try:
        s, n = play(our, opp, i % 2 == 0, 1000 + i)
        scores.append(s)
        print("game %d: %s (%d moves)" % (i + 1, "W" if s == 1 else ("L" if s == 0 else "D"), n), flush=True)
    except RuntimeError as e:
        print("game %d: ENGINE CRASH: %s" % (i + 1, e), flush=True)
        break
print("FINAL: %.2f over %d games" % (sum(scores) / max(len(scores), 1), len(scores)))
our.quit()
opp.quit()
