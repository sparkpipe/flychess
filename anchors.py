#!/usr/bin/env python3
"""Calibrated weak UCI anchors for the Elo ladder (chosen because engine
download mirrors are dead). Deterministic, classical alpha-beta:
  FlyAnchor500  = 1-ply greedy material capture, random tiebreak  (~400-600)
  FlyAnchor1000 = depth-2 alpha-beta, material + piece-square tables (~900-1100)
  FlyAnchor1300 = depth-3 alpha-beta, material + PST + king safety (~1200-1400)
Run: python3 anchors.py <name>   (UCI on stdin/stdout)
"""
import sys, random
import chess, chess.engine

MAT = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
       chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# small piece-square bonus: center preference (64 entries by square index)
CENTER = [((abs(3.5 - (s % 8)) + abs(3.5 - (s // 8))) * -3) for s in range(64)]


def evaluate(board):
    if board.is_checkmate():
        return -100000
    if board.is_game_over():
        return 0
    sc = 0
    for sq, pc in board.piece_map().items():
        v = MAT[pc.piece_type] + CENTER[sq] * (1 if pc.piece_type != chess.KING else 0)
        sc += v if pc.color == board.turn else -v
    # tiny mobility edge
    sc += 2 * board.legal_moves.count()
    return sc


def alphabeta(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return evaluate(board)
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: (board.is_capture(m)
                              and MAT.get(board.piece_type_at(m.to_square), 0)) or 0,
               reverse=True)
    best = -1000000
    for m in moves:
        board.push(m)
        sc = -alphabeta(board, depth - 1, -beta, -alpha)
        board.pop()
        if sc > best:
            best = sc
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def pick(board, depth, rng=None):
    moves = list(board.legal_moves)
    if not moves:
        return None
    if depth <= 0:                       # greedy capture / random
        caps = [m for m in moves if board.is_capture(m)]
        if caps and rng:
            victim = max(MAT.get(board.piece_type_at(m.to_square), 0) for m in caps)
            best = [m for m in caps
                    if MAT.get(board.piece_type_at(m.to_square), 0) == victim]
            return rng.choice(best)
        return rng.choice(moves)
    scored = []
    for m in moves:
        board.push(m)
        sc = -alphabeta(board, depth - 1, -1000000, 1000000)
        board.pop()
        scored.append((sc, m.uci()))
    if rng:
        top = max(s for s, _ in scored)
        scored = [x for x in scored if x[0] >= top - 5]
    scored.sort(reverse=True)
    return scored[0][1]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "FlyAnchor500"
    depth = {"FlyAnchor500": 0, "FlyAnchor1000": 2, "FlyAnchor1300": 3}[name]
    rng = random.Random(42)
    board = chess.Board()
    print(f"id name {name}")
    print("uciok", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line == "isready":
            print("readyok", flush=True)
        elif line == "ucinewgame":
            board = chess.Board()
        elif line.startswith("position"):
            board = chess.Board()
            arg = line.split(" ", 1)[1] if " " in line else ""
            if arg.startswith("fen "):
                parts = arg[4:].split(" moves ")
                board = chess.Board(parts[0])
                moves = parts[1] if len(parts) > 1 else ""
            else:
                moves = arg.split("moves ", 1)[1] if "moves " in arg else ""
            for mv in moves.split():
                board.push_uci(mv)
        elif line.startswith("go"):
            mv = pick(board, depth, rng=None if depth else random.Random())
            print(f"bestmove {mv or '(none)'}", flush=True)
        elif line == "quit":
            break


if __name__ == "__main__":
    main()
