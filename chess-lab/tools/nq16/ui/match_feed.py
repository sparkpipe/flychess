#!/usr/bin/env python3
"""Live feed v2 — MATCH HISTORY: scans the matches dir for *.pgn (one per match),
builds games.json as a LIST of matches (oldest first, newest last). Past matches
and their eval fields are never removed; new matches appear as they finish.
Eval fields are merged from the previous games.json by (tag, game identity)."""
import glob
import json
import os
import time

import chess
import chess.pgn

MDIR = "/srv/workspace/chess-active/matches"
OUT = f"{MDIR}/games.json"
TAGS = {
    "nq_vs_n16_1s.pgn": "nQ 1s vs n16 1.14s (nps-calibrated)",
    "n16quick_vs_nq_2v1.pgn": "n16 2s vs nQ 1s (10-min nets)",
}
# 3open.epd book positions -> (sidebar label, book moves from startpos)
BOOK = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w": ("startpos", []),
    "rnbqkbnr/ppp1pppp/8/3p4/3PP3/8/PPP2PPP/RNBQKBNR b":
        ("BDG (1.d4 d5 2.e4)", ["d2d4", "d7d5", "e2e4"]),
    "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b":
        ("French Advance (3.e5)", ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5"]),
}
EVAL_FIELDS = ("evals_d12", "evals_d20", "evals_d25",
               "pv_d12", "pv_d20", "pv_d25")


def book_san_for(fen0):
    start = " ".join(fen0.split()[:2])
    label, ucis = BOOK.get(start, ("book position", []))
    if not ucis:
        return label, []
    b = chess.Board()
    sans = []
    try:
        for u in ucis:
            sans.append(b.san(chess.Move.from_uci(u)))
            b.push(chess.Move.from_uci(u))
    except Exception:
        return label, []
    if b.fen().split()[0] != fen0.split()[0]:
        return label, []
    return label, sans


def parse_pgn(path):
    games = []
    with open(path) as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            board = g.board()
            sans, ucis, fens = [], [], [board.fen()]
            for mv in g.mainline_moves():
                sans.append(board.san(mv))
                ucis.append(mv.uci())
                board.push(mv)
                fens.append(board.fen())
            h = g.headers
            label, book_san = book_san_for(fens[0])
            games.append({
                "white": h.get("White", "?"), "black": h.get("Black", "?"),
                "result": h.get("Result", "*"), "ply": len(sans),
                "termination": h.get("Termination", "normal"),
                "open": h.get("Result", "*") == "*",
                "opening": label, "book_san": book_san,
                "white_elo": "", "black_elo": "",
                "date": h.get("Date", ""), "time": h.get("GameEndTime", ""),
                "sans": sans, "ucis": ucis, "fens": fens,
            })
    return games


def load_old():
    """eval fields from the previous games.json, keyed (tag, white, black, date, ply)."""
    if not os.path.exists(OUT):
        return {}
    try:
        old = json.load(open(OUT))
    except Exception:
        return {}
    keep = {}
    for m in old:
        for g in m.get("games", []):
            if any(f in g for f in EVAL_FIELDS):
                keep[(m["tag"], g["white"], g["black"], g["date"], g["ply"])] = \
                    {f: g[f] for f in EVAL_FIELDS if f in g}
    return keep


def main():
    while True:
        try:
            pgns = sorted(glob.glob(f"{MDIR}/*.pgn"), key=os.path.getmtime)
            if pgns:
                old = load_old()
                doc = []
                for p in pgns:
                    tag = TAGS.get(os.path.basename(p), os.path.basename(p)[:-4])
                    games = parse_pgn(p)
                    if not games:
                        continue
                    for g in games:
                        k = (tag, g["white"], g["black"], g["date"], g["ply"])
                        if k in old:
                            g.update(old[k])
                    doc.append({"tag": tag, "games": games})
                if doc:
                    tmp = OUT + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(doc, f)
                    os.replace(tmp, OUT)
        except Exception as e:
            print("feed error:", e, flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
