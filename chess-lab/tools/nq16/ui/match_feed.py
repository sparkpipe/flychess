#!/usr/bin/env python3
"""Live feed: parse fastchess match PGN -> games.json for the :8077 UI.

Idempotent merge: preserves eval fields already filled by match_eval_worker.
Loops every 10s; games appear as fastchess finalizes them in the PGN.
"""
import json
import os
import time

import chess
import chess.pgn

PGN = "/srv/workspace/chess-active/matches/nq_vs_n16_1s.pgn"
OUT = "/srv/workspace/chess-active/matches/games.json"
TAG = "nQ 1s vs n16 1.14s (nps-calibrated)"
EVAL_FIELDS = ("evals_d12", "evals_d20", "evals_d25",
               "pv_d12", "pv_d20", "pv_d25")
# 3open.epd book positions -> sidebar labels (keyed on board+stm fields)
BOOK = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w": "startpos",
    "rnbqkbnr/ppp1pppp/8/3p4/3PP3/8/PPP2PPP/RNBQKBNR b": "BDG (1.d4 d5 2.e4)",
    "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b": "French Advance (3.e5)",
}


def parse_pgn():
    games = []
    with open(PGN) as f:
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
            start = " ".join(fens[0].split()[:2])
            games.append({
                "white": h.get("White", "?"), "black": h.get("Black", "?"),
                "result": h.get("Result", "*"), "ply": len(sans),
                "termination": h.get("Termination", "normal"),
                "open": h.get("Result", "*") == "*",
                "opening": BOOK.get(start, "book position" if fens[0].split()[0] != "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" else "startpos"),
                "white_elo": "", "black_elo": "",
                "date": h.get("Date", ""), "time": h.get("GameEndTime", ""),
                "sans": sans, "ucis": ucis, "fens": fens,
            })
    return games


def load_old():
    if not os.path.exists(OUT):
        return {}
    try:
        doc = json.load(open(OUT))
        for m in doc:
            if m["tag"] == TAG:
                return {(g["white"], g["black"], g["date"], g["ply"]): g
                        for g in m["games"]}
    except Exception:
        pass
    return {}


def main():
    while True:
        try:
            if os.path.exists(PGN):
                games = parse_pgn()
                if games:
                    old = load_old()
                    for g in games:
                        k = (g["white"], g["black"], g["date"], g["ply"])
                        if k in old:
                            for f in EVAL_FIELDS:
                                if f in old[k]:
                                    g[f] = old[k][f]
                    doc = [{"tag": TAG, "games": games}]
                    tmp = OUT + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(doc, f)
                    os.replace(tmp, OUT)
        except Exception as e:
            print("feed error:", e, flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
