"""Opening book for arena matches: 1000 EPD positions sampled from
real OTB games at plies 10-16 (both colors to move)."""
import sys, random
import chess
import chess.pgn

rng = random.Random(20260929)
PGN = "/home/spec/chess-lab/gambit/filtered.pgn"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/cold-raid6/chess-audit/otb_book.epd"
n = 0
with open(PGN, encoding="utf-8", errors="replace") as f, open(OUT, "w") as w:
    while n < 1000:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        if rng.random() > 0.02:   # thin out
            continue
        board = g.board()
        plies = []
        for node in g.mainline():
            board.push(node.move)
            plies.append(board.fen())
            if len(plies) > 16:
                break
        if len(plies) < 10:
            continue
        p = rng.randrange(9, min(16, len(plies)))
        b = chess.Board(plies[p])
        w.write("%s bm -;\n" % b.epd())
        n += 1
print("BOOK:", n, "positions")
