import json, sys, re, chess.pgn, io, subprocess, time
pgn_path, gi = sys.argv[1], int(sys.argv[2])
txt = open(pgn_path).read()
games = txt.split("[Event")[1:]
gtxt = games[gi]
g = chess.pgn.read_game(io.StringIO("[Event" + gtxt))
wh = g.headers["White"]
v5_white = wh.startswith("v5")
res = g.headers["Result"]
print(f"game {gi+1}: v5 as {"White" if v5_white else "Black"} | result {res} | ply {len(list(g.mainline()))}")
moves = [n.move.uci() for n in g.mainline()]
# analyze with SF17 depth 18
p = subprocess.Popen(["/usr/games/stockfish"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\nisready\n"); p.stdin.flush()
while "readyok" not in p.stdout.readline(): pass
board = g.board()
worst = []
prev_cp = None
for k, mv in enumerate(moves):
    p.stdin.write(f"position fen {board.fen()}\ngo depth 18\n"); p.stdin.flush()
    cp = None
    for _ in range(4000):
        line = p.stdout.readline()
        m = re.search(r"depth 18 .*score cp (-?\d+)", line)
        if m: cp = int(m.group(1))
        if line.startswith("bestmove"): break
    stm = 1 if board.turn else -1
    cp_white = cp * stm if cp is not None else None
    if prev_cp is not None and cp_white is not None:
        mover_stm = 1 if k % 2 == 0 else -1
        loss = prev_cp * (-mover_stm) - cp_white * (-mover_stm)
        # self-harm: eval from mover persp before vs after
        loss = (prev_cp * -mover_stm) - (cp_white * -mover_stm)
        worst.append((loss, k, mv, board.fen()))
    prev_cp = cp_white
    board.push(chess.Move.from_uci(mv))
p.stdin.write("quit\n"); p.stdin.flush()
worst.sort(reverse=True)
for label, is_v5 in [("v5", v5_white), ("opp", not v5_white)]:
    side_moves = [(L, k, mv, fen) for L, k, mv, fen in worst if (k % 2 == 0) == is_v5]
    total = sum(x[0] for x in side_moves); cnt = max(len(side_moves), 1)
    bl = [x for x in side_moves if x[0] >= 100]
    print(f"  {label}: avg {total/cnt/100:.2f} pn/move, blunders {len(bl)}")
    for L, k, mv, fen in bl[:3]:
        dots = "..." if k % 2 else "."
        print(f"    {k//2+1}{dots} {mv} -{L/100:.1f}pn")
n = len(moves)
v5sign = 1 if v5_white else -1
for phase, lo, hi in [("open", 0, n//3), ("mid", n//3, 2*n//3), ("end", 2*n//3, n)]:
    print(f"  {phase}: (see blunder profile above)")
