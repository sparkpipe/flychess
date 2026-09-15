#!/usr/bin/env python3
"""Tournament-player checklist — the gate before ANY Elo measurement.

Tasks:
  A  mateIn1 conversion      >= 95%   (100 held-out puzzles, first move)
  B  mateIn2 full line       >= 50%   (100 held-out puzzles, every solver move)
  C  KQvK conversion         >= 90%   (20 random won endings vs greedy defender)
  D  KRvK conversion         >= 80%   (20 random won endings vs greedy defender)
  E  draw holding            >= 60%   (theoretical draws vs depth-3 attacker)
  F  opening sanity          >= 75%   (gambit lines, no material dump by ply 16)

Usage: python3 checklist.py [task letters, default all]
Exit 0 only when every requested gate passes. Sims fixed at 400 (match-like).
Endgame resistance = real Stockfish (defender: depth 10; attacker: 300ms/move).
Gates: mateIn1 100%, KQvK/KRvK conversion 100%, draw-hold 100%.
"""
import chess.engine

SF = "/usr/games/stockfish"


def sf_engine():
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 64})
    return eng
import sys, random, chess
import numpy as np

sys.path.insert(0, "/home/spec/chess-lab")
import fly_mcts
import anchors

PUZ = "/home/spec/chess-lab/puzzles.csv"
FLY_SIMS = 400
CONV_MOVE_CAP = 60
HOLD_MOVE_CAP = 50


def fly_move(board, seed=0):
    mv, _ = fly_mcts.bestmove(board, sims=FLY_SIMS, swarm_seed=seed)
    return mv


def greedy_move(board):
    mv = anchors.pick(board, 0)
    return chess.Move.from_uci(mv) if mv else None


def d3_move(board):
    mv = anchors.pick(board, 3)
    return chess.Move.from_uci(mv) if mv else None


def mat_balance(board, color):
    V = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
         chess.QUEEN: 9}
    s = 0
    for sq, pc in board.piece_map().items():
        v = V[pc.piece_type]
        s += v if pc.color == color else -v
    return s


def sample_puzzles(theme, n, seed=3):
    import pandas as pd
    df = pd.read_csv(PUZ, usecols=["FEN", "Moves", "Themes"])
    df = df[df.Themes.fillna("").str.contains(theme)]
    df = df.sample(n=min(n * 3, len(df)), random_state=seed)
    out = []
    for fen, moves in zip(df.FEN.values, df.Moves.values):
        try:
            b = chess.Board(fen)
            ms = moves.split()
            b.push(chess.Move.from_uci(ms[0]))
            solver = [chess.Move.from_uci(ms[i]) for i in range(1, len(ms), 2)]
            if all(m in b.legal_moves for m in solver[:1]):
                out.append((fen, ms))
        except Exception:
            continue
        if len(out) >= n:
            break
    return out


def task_a(n=100):
    """mateIn1: play the mating move."""
    ok = tot = 0
    for fen, ms in sample_puzzles("mateIn1", n):
        b = chess.Board(fen)
        b.push(chess.Move.from_uci(ms[0]))   # opponent setup move
        want = ms[1]
        mv = fly_move(b)
        tot += 1
        ok += int(mv == want)
    return ok / max(tot, 1), tot


def task_b(n=100):
    """mateIn2: full-line argmax correctness (all solver moves)."""
    ok = tot = 0
    for fen, ms in sample_puzzles("mateIn2", n):
        try:
            b = chess.Board(fen)
            b.push(chess.Move.from_uci(ms[0]))   # opponent setup move
            good = True
            for i in range(1, len(ms)):
                mv = chess.Move.from_uci(ms[i])
                if mv not in b.legal_moves:
                    good = False
                    break
                if i % 2 == 1:
                    got = fly_move(b)
                    if got != mv.uci():
                        good = False
                        break
                b.push(mv)
        except Exception:
            good = False
        tot += 1
        ok += int(good)
    return ok / max(tot, 1), tot


def won_endgame_positions(kind, n, seed=9):
    rng = random.Random(seed)
    out = []
    tries = 0
    while len(out) < n and tries < n * 30:
        tries += 1
        b = chess.Board(None)
        sqs = rng.sample(range(64), 3)
        b.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
        pt = chess.QUEEN if kind == "KQ" else chess.ROOK
        b.set_piece_at(sqs[2], chess.Piece(pt, chess.WHITE))
        if not b.is_valid() or b.is_game_over():
            continue
        b.turn = rng.random() < 0.5
        if b.is_game_over():
            continue
        out.append(b)
    return out


def task_cd(kind, n=20):
    """Conversion: fly mates STOCKFISH (depth-10 defender) within cap moves.
    Fly plays whichever side owns the heavy piece."""
    eng = sf_engine()
    conv = 0
    try:
        for b0 in won_endgame_positions(kind, n):
            b = b0.copy()
            fly_is_white = (b.turn == chess.WHITE)
            moves = 0
            while not b.is_game_over(claim_draw=True) and moves < CONV_MOVE_CAP * 2:
                if (b.turn == chess.WHITE) == fly_is_white:
                    mv = fly_move(b, seed=moves)
                    if mv is None:
                        break
                    b.push_uci(mv)
                else:
                    r = eng.play(b, chess.engine.Limit(depth=10))
                    b.push(r.move)
                moves += 1
            oc = b.outcome(claim_draw=False)
            if oc and oc.winner is not None:
                fly_won = (oc.winner == chess.WHITE) == fly_is_white
                conv += int(fly_won)
    finally:
        eng.quit()
    return conv / n, n


HOLD_FENS = [
    "k7/8/K7/1P6/8/6r1/8/6R1 w - - 0 1",     # Philidor 3rd-rank defense
    "k7/8/K7/P7/8/6r1/8/6R1 w - - 0 1",      # Philidor with a5
    "rk6/pb6/1K6/8/8/8/6R1/8 w - - 0 1",     # R vs R+B corner fortress
    "1k6/8/8/8/8/2nb4/8/1K1R4 w - - 0 1",    # exchange-down-ish block
    "8/8/8/4k3/8/8/8/K7 w - - 0 1",          # KvK dead draw
]


def task_e():
    """Draw holding: defend theoretical draws vs STOCKFISH (300ms/move, which
    grinds real wins) — hold to the move cap or draw. 100% gate."""
    eng = sf_engine()
    held = 0
    try:
        for fen in HOLD_FENS:
            b = chess.Board(fen)
            moves = 0
            fly_is_white = b.turn == chess.WHITE
            while not b.is_game_over(claim_draw=True) and moves < HOLD_MOVE_CAP * 2:
                if (b.turn == chess.WHITE) == fly_is_white:
                    mv = fly_move(b, seed=moves)
                    if mv is None:
                        break
                    b.push_uci(mv)
                else:
                    r = eng.play(b, chess.engine.Limit(time=0.3))
                    b.push(r.move)
                moves += 1
            oc = b.outcome(claim_draw=False)
            if oc is None or oc.winner is None:
                held += 1                    # survived cap or drew: HELD
    finally:
        eng.quit()
    return held / len(HOLD_FENS), len(HOLD_FENS)


def task_f(n=20):
    """Opening sanity: from gambit-book starts, fly must not dump material
    (>= -2 pawns at ply 16) vs the depth-3 anchor."""
    import chess.pgn
    sane = tot = 0
    games = []
    with open("/home/spec/chess-lab/gambit_lines.pgn") as f:
        while len(games) < n:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            games.append(g)
    for g in games:
        b = g.board()
        path = list(g.mainline_moves())[:16]
        for i, mv in enumerate(path):
            b.push(mv)
        fly_is_white = b.turn == chess.WHITE
        while not b.is_game_over() and b.ply() < 16 + 8:
            if (b.turn == chess.WHITE) == fly_is_white:
                mv = fly_move(b, seed=b.ply())
                if mv is None:
                    break
                b.push_uci(mv)
            else:
                mv = d3_move(b)
                if mv is None:
                    break
                b.push(mv)
        bal = mat_balance(b, fly_is_white)
        tot += 1
        sane += int(bal >= -2)
    return sane / max(tot, 1), tot


GATES = {
    "A": ("mateIn1 == 100%", task_a, 1.00),
    "B": ("mateIn2 full line >= 50%", task_b, 0.50),
    "C": ("KQvK vs SF == 100%", lambda: task_cd("KQ"), 1.00),
    "D": ("KRvK vs SF == 100%", lambda: task_cd("KR"), 1.00),
    "E": ("draw hold vs SF == 100%", task_e, 1.00),
    "F": ("opening sanity >= 75%", task_f, 0.75),
}


def main():
    want = sys.argv[1:] if len(sys.argv) > 1 else list(GATES)
    all_pass = True
    print(f"{'task':4s} {'score':>7s} {'gate':>6s}  {'pass':4s} description")
    for t in want:
        name, fn, gate = GATES[t]
        sc, n = fn()
        ok = sc >= gate
        all_pass &= ok
        print(f"{t:4s} {sc:7.2f} {gate:6.2f}  {'PASS' if ok else 'FAIL':4s} {name} (n={n})",
              flush=True)
    print("CHECKLIST:", "ALL PASS — cleared for Elo" if all_pass else "NOT READY")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
