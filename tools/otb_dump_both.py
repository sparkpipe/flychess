"""OTB FULL-GAME DUMP, BOTH SIDES (ruling 2026-09-27: "the losing side can also
create training worthy data" — defensive play and exchange-down saving are the
DOWN side's data).

Identical to otb_dump.py except every position of the game is emitted
(both sides to move), with the played move of THAT position as the label.

Output format: one line per position:
  fen | played_move_uci | winner | ply | white_elo | black_elo | result

Qualifying: decisive games, winner >= 2400, loser >= 2000 (operator bar).
Pre-1930 games pass on archive-trust.
"""
import sys
import os
import chess
import chess.pgn

PGN = os.environ.get("PGN",
                     "/home/spec/chess-lab/gambit/filtered.pgn")
OUT = os.environ.get("OUT",
                     "/home/spec/chess-lab/otb_complete_dump_both.txt")
MAX_GAMES = int(os.environ.get("MAX_GAMES", "0"))  # 0 = all


def main():
    n_games = n_kept = n_pos = 0
    with open(PGN, encoding="utf-8", errors="replace") as f, \
            open(OUT, "w") as out:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            n_games += 1
            if MAX_GAMES and n_games > MAX_GAMES:
                break
            res = g.headers.get("Result", "*")
            if res not in ("1-0", "0-1"):
                continue
            try:
                wel = int(g.headers.get("WhiteElo", 0) or 0)
                bel = int(g.headers.get("BlackElo", 0) or 0)
            except Exception:
                continue
            winner_is_white = res == "1-0"
            we, le = (wel, bel) if winner_is_white else (bel, wel)

            date = g.headers.get("Date", "") or ""
            try:
                yr = int(date[:4])
            except Exception:
                yr = 0
            hist = 0 < yr <= 1930
            elo_given = not (wel == 0 and bel == 0)
            if not hist and elo_given and (we < 2400 or le < 2000):
                continue

            winner = "w" if winner_is_white else "b"
            board = g.board()
            ply = 0
            for node in g.mainline():
                ply += 1
                if ply > 300:
                    break
                fen_before = board.fen()
                u = node.move.uci()
                out.write(f"{fen_before}|{u}|{winner}|{ply}"
                          f"|{wel}|{bel}|{res}\n")
                n_pos += 1
                try:
                    board.push(node.move)
                except Exception:
                    break
            n_kept += 1
            if n_kept % 100000 == 0:
                print(f"games={n_games} kept={n_kept} "
                      f"positions={n_pos}", flush=True)
    print(f"TOTAL: {n_games} games scanned, {n_kept} kept, "
          f"{n_pos} both-side positions dumped", flush=True)
    print("OTB-DUMP-BOTH-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
