#!/usr/bin/env python3
"""MINIATURE ROWS — CORRECTED per operator 2026-10-09.

Spec: games where a 2400+ player LOSES quickly (upsets); training rows are
every WINNER-side position from move 1 to the finish.
  - loser Elo >= 2400, winner Elo >= 2000
  - decisive only
  - game qualifies if it ENDS by move 25, OR winner's material edge first
    reaches +5 by move 25 (truncate there). Games doing neither NEVER qualify.

Supersedes miniature_rows.py whose filters were inverted (strong winner) and
which never checked game length at all (49% of rows came from long games)."""
import chess
import chess.pgn

PGN = "/mnt/cold-raid6/rtx5090-archive/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn"
OUT = "/srv/workspace/chess-active/miniature_rows_v2.tsv"
LOSER_ELO = 2400   # the strong player who loses
WINNER_ELO = 2000  # the winner's floor
MAXMOVE = 25
MATCAP = 5
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def elo(h, k):
    try:
        return int(h.get(k, "0"))
    except ValueError:
        return 0


def mat_edge(b, winner):
    me = 0 if winner else 1
    opp = 1 - me
    s = 0
    for pt in VAL:
        s += VAL[pt] * (len(b.pieces(pt, me)) - len(b.pieces(pt, opp)))
    return s


games = rows = 0
qual = 0
out = open(OUT, "w")
with open(PGN, encoding="utf-8", errors="replace") as f:
    while True:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        games += 1
        h = g.headers
        res = h.get("Result", "")
        if res not in ("1-0", "0-1"):
            continue
        we = elo(h, "WhiteElo")
        be = elo(h, "BlackElo")
        # 2400+ LOSER, 2000+ winner
        if res == "1-0" and be >= LOSER_ELO and we >= WINNER_ELO:
            loser, winner = chess.BLACK, chess.WHITE
        elif res == "0-1" and we >= LOSER_ELO and be >= WINNER_ELO:
            loser, winner = chess.WHITE, chess.BLACK
        else:
            continue
        try:
            b = g.board()
        except Exception:
            continue
        if b is None:
            continue
        moves = list(g.mainline_moves())
        total_plies = len(moves)
        # qualification: ends by move 25 (<= 49 plies) OR winner edge >= +5 by move 25
        ends_by_25 = total_plies <= 2 * MAXMOVE - 1
        truncate_at = None
        bb = b.copy()
        for i, mv in enumerate(moves):
            bb.push(mv)
            if bb.fullmove_number <= MAXMOVE and mat_edge(bb, winner) >= MATCAP:
                truncate_at = i + 1
                break
        if not ends_by_25 and truncate_at is None:
            continue  # leaked long grind: NEVER qualifies
        limit = truncate_at if truncate_at is not None else total_plies
        qual += 1
        gid = qual
        i = 0
        for mv in moves[:limit]:
            if b.turn == winner:
                out.write(f"{b.fen()}\t{mv.uci()}\t{gid}\t{i}\n")
                rows += 1
            b.push(mv)
            i += 1
        if games % 200000 == 0:
            print(f"{games:,} games scanned, {qual:,} qualified, {rows:,} rows",
                  flush=True)
out.close()
print(f"DONE: {games:,} games scanned, {qual:,} qualifying miniatures, {rows:,} rows")
