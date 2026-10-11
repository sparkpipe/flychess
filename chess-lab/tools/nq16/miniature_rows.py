"""MINIATURE ROWS — operator spec 2026-10-05: NO SF grading/filtering.
Every winner-side position from move 1 to the finish of qualifying miniatures.

Filters (operator directive 2026-09-25): winner Elo >= 2400, loser >= 2000,
decisive, finish by move 25 (game ends <= move 25 OR winner's material edge
first reaches +5 by move 25 — truncate there).

Rows: winner-to-move positions only: fen, played_uci, gid, ply.
Labels: d12 cp attached in a second pass (same convention as all sources).
Output: /extnvme/active/miniature_rows.tsv
"""
import chess, chess.pgn

PGN = "/mnt/cold-raid6/rtx5090-archive/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn"
OUT = "/extnvme/active/miniature_rows.tsv"
WIN_ELO = 2400
LOSE_ELO = 2000
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
out = open(OUT, "w")
with open(PGN, encoding="utf-8", errors="replace") as f:
    while True:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        h = g.headers
        res = h.get("Result", "")
        if res not in ("1-0", "0-1"):
            continue
        we = elo(h, "WhiteElo")
        be = elo(h, "BlackElo")
        if res == "1-0" and we >= WIN_ELO and be >= LOSE_ELO:
            winner = chess.WHITE
        elif res == "0-1" and be >= WIN_ELO and we >= LOSE_ELO:
            winner = chess.BLACK
        else:
            continue
        try:
            b = g.board()
        except Exception:
            continue
        if b is None:
            continue
        gid = games
        done = False
        i = 0
        for mv in g.mainline_moves():
            if done:
                break
            if b.turn == winner:
                out.write(f"{b.fen()}\t{mv.uci()}\t{gid}\t{i}\n")
                rows += 1
            b.push(mv)
            i += 1
            if b.fullmove_number <= MAXMOVE and mat_edge(b, winner) >= MATCAP:
                done = True   # winner sealed it; rest of game ignored
        games += 1
        if games % 500 == 0:
            print(f"{games} miniatures, {rows} rows", flush=True)
out.close()
print(f"DONE {games} miniatures, {rows} rows")
