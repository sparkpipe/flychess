"""Extract positions from a self-play PGN for training.
Output: fen|move|winner|ply|0|0|gameid (eval-fleet compatible format)."""
import sys
import chess
import chess.pgn

pgn_path, out_path = sys.argv[1], sys.argv[2]
n_pos = n_games = 0
with open(pgn_path) as f, open(out_path, "w") as w:
    while True:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        n_games += 1
        gid = g.headers.get("GameId", str(n_games))
        board = g.board()
        ply = 0
        for node in g.mainline():
            fen = board.fen()
            mv = node.move.uci()
            res = g.headers.get("Result", "*")
            winner = "w" if res == "1-0" else "b" if res == "0-1" else "d"
            w.write("%s|%s|%s|%d|0|0|%s\n" % (fen, mv, winner, ply + 1, gid))
            n_pos += 1
            board.push(node.move)
            ply += 1
print("SELFPLAY EXTRACT: %d games -> %d positions" % (n_games, n_pos))
