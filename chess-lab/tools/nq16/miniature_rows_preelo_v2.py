#!/usr/bin/env python3
"""Pre-Elo miniature pass — operator-approved name list (2026-10-09).
Winner must be a recognized pre-rating master (list below, surname match),
loser unrestricted, correspondence EXCLUDED, same length law as Elo pass
(ends by move 25 OR winner +5 material edge by move 25, truncate).
Prints per-name database match counts for verification before anything scales."""
import chess
import chess.pgn

PGN = "/mnt/cold-raid6/rtx5090-archive/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn"
CENSUS = "/srv/workspace/chess-active/players_census.tsv"
OUT = "/srv/workspace/chess-active/miniature_rows_preelo.tsv"
MAXMOVE, MATCAP = 25, 5

NAMES = [
    "Steinitz", "Lasker", "Capablanca", "Alekhine", "Euwe", "Botvinnik",
    "Smyslov", "Tal", "Petrosian", "Spassky",
    "Morphy", "Anderssen", "Blackburne", "Zukertort", "Chigorin", "Tarrasch",
    "Marshall", "Pillsbury", "Schlechter", "Janowski", "Showalter", "Gunsberg",
    "Maroczy", "Burn", "Mason", "Winawer", "Teichmann", "Duras", "Leonhardt",
    "Spielmann",
    "Rubinstein", "Nimzowitsch", "Tartakower", "Reti", "Grünfeld", "Grünfeld",
    "Bogoljubov", "Flohr", "Fine", "Reshevsky", "Vidmar", "Kostic", "Colle",
    "Kmoch", "Yates", "Thomas", "Sämisch", "Samisch", "Ahues", "Keres",
    "Stoltz", "Alexander",
    "Bronstein", "Geller", "Boleslavsky", "Lilienthal", "Najdorf", "Stahlberg",
    "Szabo", "Kotov", "Averbakh", "Taimanov", "Gligoric", "Unzicker",
    "Olafsson", "Larsen", "Ivkov", "Darga", "Panno", "Trifunovic", "Barcza",
    "Bondarevsky",
]
NAMES = sorted(set(NAMES))

# census: surname (before comma) -> total games, for match-count verification
census_games = {}
for line in open(CENSUS, encoding="utf-8", errors="replace"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        full = p[0]
        surname = full.split(",")[0].strip()
        try:
            g = int(p[1])
        except ValueError:
            continue
        census_games[surname] = census_games.get(surname, 0) + g

print("=== list-name database match counts (games) ===")
for n in NAMES:
    print(f"  {n:<14} {census_games.get(n, 0):>7,}")
NAMESET = set(NAMES)

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def surname(name):
    return name.split(",")[0].strip()


def mat_edge(b, winner):
    me = 0 if winner else 1
    opp = 1 - me
    return sum(VAL[pt] * (len(b.pieces(pt, me)) - len(b.pieces(pt, opp)))
               for pt in VAL)


def has_elo(h):
    # game is Elo-era only if BOTH headers EXIST and parse; missing header
    # must NOT default to a parseable value (that bug classified all pre-Elo
    # games as Elo-era and yielded zero qualifiers)
    for k in ("WhiteElo", "BlackElo"):
        v = h.get(k)
        if v is None:
            return False
        try:
            int(v)
        except ValueError:
            return False
    return True


out = open(OUT, "w")
qual = rows = 0
examples = 0
with open(PGN, encoding="utf-8", errors="replace") as f:
    while True:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        h = g.headers
        res = h.get("Result", "")
        if res not in ("1-0", "0-1") or has_elo(h):
            continue
        if "corr" in h.get("Event", "").lower():
            continue
        # era cut: "before ELO existed" — no-Elo headers alone also match modern
        # club games, so require a dated game from the pre-rating epoch
        try:
            year = int(h.get("Date", "????.??.??")[:4])
        except ValueError:
            continue
        if year > 1972:
            continue
        winner = chess.WHITE if res == "1-0" else chess.BLACK
        wname = h.get("White" if winner == chess.WHITE else "Black", "?")
        if surname(wname) not in NAMESET:
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
        if examples < 10:
            lname = h.get("Black" if winner == chess.WHITE else "White", "?")
            print(f"  e.g. {wname} beat {lname} — {res}, rows to ply {limit}",
                  flush=True)
            examples += 1
        gid = 2_000_000 + qual
        i = 0
        for mv in moves[:limit]:
            if b.turn == winner:
                out.write(f"{b.fen()}\t{mv.uci()}\t{gid}\t{i}\n")
                rows += 1
            b.push(mv)
            i += 1
out.close()
print(f"PRE-ELO DONE: {qual:,} qualifying games, {rows:,} rows -> {OUT}")
