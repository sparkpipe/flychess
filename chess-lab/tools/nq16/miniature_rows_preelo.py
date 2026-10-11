#!/usr/bin/env python3
"""Pre-Elo miniature pass (operator 2026-10-09): games from before ratings,
included when the WINNER was a recognized player (PGN title OR >=100 games in
the database per players_census.tsv). Same length law: ends by move 25 OR
winner's +5 material edge by move 25 (truncate). Rows: winner-side positions.
Usage: miniature_rows_preelo.py <sample_only|build>"""
import sys
import chess
import chess.pgn

PGN = "/mnt/cold-raid6/rtx5090-archive/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn"
CENSUS = "/srv/workspace/chess-active/players_census.tsv"
OUT = "/srv/workspace/chess-active/miniature_rows_preelo.tsv"
MODE = sys.argv[1] if len(sys.argv) > 1 else "build"
MAXMOVE, MATCAP = 25, 5
MIN_GAMES = 100
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

rec = {}
for line in open(CENSUS, encoding="utf-8", errors="replace"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        rec[p[0]] = (int(p[1]), p[2] if len(p) > 2 else "")


def recognized(name):
    r = rec.get(name)
    return r is not None and (r[0] >= MIN_GAMES or bool(r[1]))


def mat_edge(b, winner):
    me = 0 if winner else 1
    opp = 1 - me
    return sum(VAL[pt] * (len(b.pieces(pt, me)) - len(b.pieces(pt, opp)))
               for pt in VAL)


def no_elo(h):
    try:
        int(h.get("WhiteElo", "0"))
        int(h.get("BlackElo", "0"))
        return False  # has Elo -> belongs to the Elo-era pass
    except (ValueError, TypeError):
        return True


out = open(OUT, "w") if MODE == "build" else None
qual = rows = shown = 0
with open(PGN, encoding="utf-8", errors="replace") as f:
    while True:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        h = g.headers
        res = h.get("Result", "")
        if res not in ("1-0", "0-1") or not no_elo(h):
            continue
        winner = chess.WHITE if res == "1-0" else chess.BLACK
        wname = h.get("White" if winner == chess.WHITE else "Black", "?")
        if not recognized(wname):
            continue
        try:
            b = g.board()
        except Exception:
            continue
        moves = list(g.mainline_moves())
        total = len(moves)
        ends_by_25 = total <= 2 * MAXMOVE - 1
        trunc = None
        bb = b.copy()
        for i, mv in enumerate(moves):
            bb.push(mv)
            if bb.fullmove_number <= MAXMOVE and mat_edge(bb, winner) >= MATCAP:
                trunc = i + 1
                break
        if not ends_by_25 and trunc is None:
            continue
        limit = trunc if trunc is not None else total
        qual += 1
        if MODE == "sample" and shown < 12:
            lname = h.get("Black" if winner == chess.WHITE else "White", "?")
            why = "ended" if ends_by_25 else "edge+5 truncated"
            r = rec.get(wname, (0, ""))
            print(f"{wname} ({r[0]} games{',' + r[1] if r[1] else ''}) beat "
                  f"{lname} — {res}, rows to ply {limit} [{why}]")
            shown += 1
        if MODE == "build":
            gid = 1_000_000 + qual  # distinct id space from Elo-era pass
            i = 0
            for mv in moves[:limit]:
                if b.turn == winner:
                    out.write(f"{b.fen()}\t{mv.uci()}\t{gid}\t{i}\n")
                    rows += 1
                b.push(mv)
                i += 1
print(f"pre-Elo qualifying: {qual:,} rows: {rows:,}")
