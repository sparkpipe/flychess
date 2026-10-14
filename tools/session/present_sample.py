#!/usr/bin/env python3
"""Present the first N qualifying games (corrected filter) with names/Elo/length
for operator approval — independent re-derivation, not trusting the builder."""
import chess
import chess.pgn

PGN = "/mnt/cold-raid6/rtx5090-archive/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn"
N = 12
LOSER_ELO, WINNER_ELO, MAXMOVE, MATCAP = 2400, 2000, 25, 5
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def elo(h, k):
    try:
        return int(h.get(k, "0"))
    except ValueError:
        return 0


def mat_edge(b, winner):
    me = 0 if winner else 1
    opp = 1 - me
    return sum(VAL[pt] * (len(b.pieces(pt, me)) - len(b.pieces(pt, opp)))
               for pt in VAL)


shown = 0
with open(PGN, encoding="utf-8", errors="replace") as f:
    while shown < N:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        h = g.headers
        res = h.get("Result", "")
        if res not in ("1-0", "0-1"):
            continue
        we, be = elo(h, "WhiteElo"), elo(h, "BlackElo")
        if res == "1-0" and be >= LOSER_ELO and we >= WINNER_ELO:
            winner = chess.WHITE
        elif res == "0-1" and we >= LOSER_ELO and be >= WINNER_ELO:
            winner = chess.BLACK
        else:
            continue
        moves = list(g.mainline_moves())
        total = len(moves)
        ends_by_25 = total <= 2 * MAXMOVE - 1
        trunc = None
        b = g.board()
        for i, mv in enumerate(moves):
            b.push(mv)
            if b.fullmove_number <= MAXMOVE and mat_edge(b, winner) >= MATCAP:
                trunc = i + 1
                break
        if not ends_by_25 and trunc is None:
            continue
        limit = trunc if trunc is not None else total
        why = f"ended move {(total + 1) // 2}" if ends_by_25 else f"edge+5 at move {(trunc + 1) // 2}, truncated"
        wname, lname = (h.get("White", "?"), h.get("Black", "?")) if winner == chess.WHITE \
            else (h.get("Black", "?"), h.get("White", "?"))
        welo, lelo = (we, be) if winner == chess.WHITE else (be, we)
        print(f"{wname} ({welo}) beat {lname} ({lelo}) — {res}, rows to ply {limit} [{why}]")
        shown += 1
