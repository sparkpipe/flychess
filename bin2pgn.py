#!/usr/bin/env python3
"""Convert a polyglot gambit book (Daring.bin) into a PGN of gambit lines:
DFS from startpos along book moves, emitting each path to a leaf/depth cap
as one PGN game. fastchess replays these lines, engines continue after."""
import chess, chess.polyglot, sys

SRC = "/home/spec/chess-lab/Daring.bin"
OUT = "/home/spec/chess-lab/gambit_lines.pgn"
MAX_PLIES = 24
MIN_W = 2            # skip weight-1 noise entries
MAX_LINES = 20000

lines = []


def dfs(board, moves):
    if len(moves) >= MAX_PLIES:
        lines.append(list(moves))
        return
    with chess.polyglot.open_reader(SRC) as r:
        entries = [e for e in r.find_all(board) if e.weight >= MIN_W]
    if not entries:
        if moves:
            lines.append(list(moves))
        return
    if len(entries) > 1 or True:
        extended = False
        for e in entries:
            mv = e.move
            if mv in board.legal_moves:
                board.push(mv)
                moves.append(mv)
                dfs(board, moves)
                moves.pop()
                board.pop()
                extended = True
                if len(lines) >= MAX_LINES:
                    return
        if not extended and moves:
            lines.append(list(moves))


def main():
    board = chess.Board()
    dfs(board, [])
    seen = set()
    n = 0
    with open(OUT, "w") as f:
        for path in lines:
            key = " ".join(m.uci() for m in path)
            if key in seen or len(path) < 4:
                continue
            seen.add(key)
            n += 1
            b = chess.Board()
            f.write('[Event "Daring gambit line"]\n[Result "*"]\n\n')
            f.write(b.variation_san(path) + " *\n\n")
    print(f"wrote {n} gambit lines -> {OUT}")


if __name__ == "__main__":
    main()
