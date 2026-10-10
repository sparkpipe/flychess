"""Eval curves: replay PGN games, evaluate every 5 moves at depth 12."""
import sys
import io
import chess
import chess.pgn
import subprocess

PGN = sys.argv[1]
games = []
with open(PGN) as f:
    while True:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        games.append(g)

p = subprocess.Popen(["/usr/games/stockfish"], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                     text=True, bufsize=1)
p.stdin.write("uci\n")
while not p.stdout.readline().startswith("uciok"):
    pass
p.stdin.write("setoption name Threads value 1\nisready\n")
while not p.stdout.readline().startswith("readyok"):
    pass

def ev(fen):
    p.stdin.write("position fen %s\ngo depth 12\n" % fen)
    p.stdin.flush()
    cp = 0
    while True:
        r = p.stdout.readline()
        if r.startswith("info ") and " score cp " in r:
            t = r.split()
            cp = int(t[t.index("cp") + 1])
        elif r.startswith("bestmove"):
            return cp

for gi, g in enumerate(games):
    board = g.board()
    moves = [n.move for n in g.mainline()]
    print("Game %d: %s (white) vs %s (black) -> %s"
          % (gi + 1, g.headers["White"], g.headers["Black"],
             g.headers["Result"]))
    print("  move  eval(cp, white persp)")
    for mi in range(0, len(moves), 10):   # every 5 full moves
        fen = board.fen()
        c = ev(fen)
        print("  %5d  %+d" % (mi // 2 + 1, c))
        for mv in moves[mi:mi + 10]:
            board.push(mv)
    print()
p.stdin.write("quit\n")
