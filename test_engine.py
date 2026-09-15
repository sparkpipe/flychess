import chess, time, sys
sys.path.insert(0, "/home/spec/chess-lab")
import fly_mcts

tests = [
    ("mateIn2 tactic", chess.Board("5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27"), "d3d6"),
    ("KQvK win (white)", chess.Board("8/8/8/4k3/8/8/8/K3Q3 w - - 0 1"), None),
    ("KvK draw-in-hand", chess.Board("8/8/8/4k3/8/8/8/K7 w - - 0 1"), None),
]
for name, b, want in tests:
    t0 = time.time()
    mv, stats = fly_mcts.bestmove(b, sims=300, trap_lambda=0.5)
    dt = time.time() - t0
    mvs, T, cls = fly_mcts.fly_eval(b)
    line = f"{name}: best={mv} ({dt:.1f}s, {len(stats)} root moves) class[L/P/W]={[round(float(c), 3) for c in cls]}"
    if want:
        line += f" want={want} match={mv == want}"
    print(line, flush=True)
