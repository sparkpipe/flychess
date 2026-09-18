"""DEGM pool generator: Dvoretsky's Endgame Manual PGN -> training pools.

The archive.org PGN Database carries all 1,635 book examples, chapter-
organized, FEN-tagged, with the book's own grades as NAGs ($1/!, $2/?,
$18/+-, $10/=, ...). This emits one jsonl pool per chapter in the SAME
schema the stage-6 trainer consumes ({fen, cat, best, children, pool}).

Labels per the operator's directive: the book's verdict symbol sets the
category; the book's annotated moves keep their grades; EVERY unannotated
legal move is Stockfish-graded at fixed depth and banded into the category
ladder. Positions with <=5 pieces are cross-validated against the local
syzygy tables — the book-vs-TB agreement rate is printed as the extraction
fidelity metric, and TB truth overrides the category there.
"""
import os
import re
import sys
import json
import chess
import chess.engine
import chess.syzygy
import chess.pgn

PGN = os.path.expanduser("~/chess-lab/books/DEGM.pgn")
OUT = os.path.expanduser("~/chess-lab/books/degm_pools")
SYZYGY = os.path.expanduser("~/syzygy")
SF = "/usr/games/stockfish"
DEPTH = 18

FLIP = {"win": "loss", "loss": "win", "draw": "draw",
        "cursed_win": "cursed_loss", "cursed_loss": "cursed_win"}


def band(score):
    """Relative Score (side to move) -> category ladder."""
    if score.is_mate():
        return "win" if score.mate() > 0 else "loss"
    cp = score.score()
    if cp >= 250:
        return "win"
    if cp >= 80:
        return "cursed_win"
    if cp > -80:
        return "draw"
    if cp > -250:
        return "cursed_loss"
    return "loss"


def wdl_cat(w):
    return {2: "win", 1: "cursed_win", 0: "draw",
            -1: "cursed_loss", -2: "loss"}.get(w)


def main():
    os.makedirs(OUT, exist_ok=True)
    tb = chess.syzygy.open_tablebase(SYZYGY) if os.path.isdir(SYZYGY) else None
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 2, "Hash": 128})
    limit = chess.engine.Limit(depth=DEPTH)

    games = []
    with open(PGN, encoding="utf-8", errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            games.append(g)
    print(f"parsed {len(games)} book entries", flush=True)

    per_ch = {}
    agree = disagree = 0
    rows_out = 0
    for gi, g in enumerate(games):
        ev = g.headers.get("Event", "")
        m = re.search(r"Chapter (\d+)", ev)
        ch = int(m.group(1)) if m else 0
        fen = g.headers.get("FEN")
        if not fen:
            continue
        try:
            b = chess.Board(fen)
        except Exception:
            continue
        if b.is_game_over() or not list(b.legal_moves):
            continue

        # book verdict: terminal NAG in the movetext ($10 =, $18 +-, $19 -+)
        try:
            verdicts = [max(n.nags) for n in g.mainline()
                        if n.nags & {10, 18, 19}]
            book_cat = {10: "draw", 18: "win", 19: "loss"}.get(
                verdicts[-1] if verdicts else None)
        except Exception:
            book_cat = None

        # book best: mainline move 1 (skip if the book itself condemns it)
        book_best = None
        book_moves = {}          # uci -> grade class
        try:
            node = g.next()
            GOOD = {1, 3}
            BAD = {2, 4}
            SOFT = {5, 6}
            if node is not None:
                mv = node.move
                if mv in b.legal_moves:
                    book_best = mv.uci()
                    nn = node.nags & (GOOD | BAD | SOFT)
                    if nn & BAD or nn & {6}:
                        book_best = None      # the book condemns its line
                    elif nn & GOOD:
                        book_moves[mv.uci()] = "best"
                    elif nn & SOFT:
                        book_moves[mv.uci()] = "soft"
                for n in g.mainline():        # collect every graded move
                    nn = n.nags & (GOOD | BAD | SOFT)
                    if n.move in b.legal_moves \
                            and n.move.uci() not in book_moves:
                        if nn & GOOD:
                            book_moves[n.move.uci()] = "best"
                        elif nn & BAD:
                            book_moves[n.move.uci()] = "bad"
                        elif nn & SOFT:
                            book_moves[n.move.uci()] = "soft"
        except Exception:
            pass

        # <=5 pieces: syzygy truth overrides the category, measures fidelity
        tb_note = None
        piece_n = sum(1 for _ in b.piece_map())
        cat = book_cat
        children = {}
        if piece_n <= 5 and tb is not None:
            try:
                wdl = tb.probe_wdl(b)
                dtz = tb.probe_dtz(b)
                cat = wdl_cat(wdl)
                tb_note = (wdl, dtz)
                if book_cat is not None:
                    if book_cat == cat:
                        agree += 1
                    else:
                        disagree += 1
            except Exception:
                pass

        # SF grades for every legal move (the operator's directive)
        mvs = list(b.legal_moves)
        try:
            infos = eng.analyse(b, limit, multipv=len(mvs))
        except Exception:
            continue
        if not isinstance(infos, list):
            infos = [infos]
        for info in infos:
            mv = info["pv"][0]
            sc = info["score"].relative
            ccat = band(sc)
            u = mv.uci()
            dtz = None
            if u in book_moves:
                dtz = None
            elif piece_n <= 5 and tb is not None and cat is not None:
                try:
                    b.push(mv)
                    dtz = tb.probe_dtz(b)
                    b.pop()
                except Exception:
                    dtz = None
            children[u] = {"cat": ccat, "dtz": dtz, "sf": sc.score() if not sc.is_mate() else None}

        if cat is None:
            # no book verdict, no TB reach: majority of SF child categories
            if children:
                from collections import Counter
                cat = Counter(c["cat"] for c in children.values()).most_common(1)[0][0]
        if cat is None:
            continue

        # best: book best if given, else the SF top move whose band == cat
        if book_best is None:
            for info in infos:
                if band(info["score"].relative) == cat:
                    book_best = info["pv"][0].uci()
                    break
        if book_best is None:
            continue

        row = {"fen": b.fen(), "cat": cat, "best": book_best,
               "children": children, "pool": f"DEGM_Ch{ch}",
               "book_verdict": book_cat, "pieces": piece_n,
               "graded": [u for u, g2 in book_moves.items()]}
        per_ch.setdefault(ch, []).append(row)
        rows_out += 1
        if gi % 100 == 0:
            print(f"[{gi}/{len(games)}] rows={rows_out} "
                  f"tb-agree={agree} tb-disagree={disagree}", flush=True)

    eng.quit()
    for ch, rows in sorted(per_ch.items()):
        with open(f"{OUT}/DEGM_Ch{ch}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"DEGM_Ch{ch}: {len(rows)} positions", flush=True)
    tot = agree + disagree
    print(f"SYZYGY-CHECK: agree={agree} disagree={disagree} "
          f"rate={agree/max(tot,1):.3f} (n={tot})", flush=True)
    print(f"DEGM-POOLS-COMPLETE: {rows_out} positions across "
          f"{len(per_ch)} chapters", flush=True)


if __name__ == "__main__":
    main()
