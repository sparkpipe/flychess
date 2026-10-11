"""Tactics expert data: dump every position of every Lichess puzzle's
winning combination. The ENTIRE combination is correct (operator ruling);
secondary solutions may also work — we pack the primary line.

Output format matches the eval-fleet input:
  fen | solution_move_uci | tactics | ply_in_line | rating | 0 | puzzle

Usage: dump_puzzle_lines.py <lichess_db_puzzle.csv> <out.txt>
"""
import sys
import csv
import chess

inp, out = sys.argv[1], sys.argv[2]
n = skipped = 0
with open(inp, newline="", encoding="utf-8") as f, open(out, "w") as w:
    for row in csv.DictReader(f):
        fen, moves = row["FEN"], row["Moves"].split()
        try:
            board = chess.Board(fen)
            for i, u in enumerate(moves):
                mv = chess.Move.from_uci(u)
                if mv not in board.legal_moves:
                    raise ValueError("illegal")
                w.write("%s|%s|tactics|%d|%s|0|%s\n"
                        % (board.fen(), u, i + 1, row["Rating"], row["PuzzleId"]))
                n += 1
                board.push(mv)
        except Exception:
            skipped += 1
            continue
print("PUZZLE LINES: %d positions, %d puzzles skipped (illegal)" % (n, skipped))
print("PUZZLE-DUMP-COMPLETE")
