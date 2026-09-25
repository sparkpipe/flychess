"""DEGM corpus rebuild (operator rulings 2026-09-19):

1. Every mainline position of every book example becomes ONE standalone
   row (no history in the fly — positions are independent questions).
2. SF grades every legal move at depth 18 (audit-proven stable), syzygy
   overrides where the tablebase reaches. Keys are self-consistent by
   construction: cat := best achievable band, tolerance set := top band.
3. BOOK BOOST: when the book's move is in the top band it becomes
   `best` (the CE primary target); otherwise SF's top move is best and
   the book move is kept as a tag only — no more contradictory keys.
4. Illegal/unreadable positions are dropped at the door.

children cat convention = band from the PARENT MOVER's perspective of
the line after that move (same convention the trainer's graded_targets
flip expects).

Output: tbpools/DEGM2_Ch{n}.jsonl + degm2_manifest.json
"""
import os
import re
import sys
import json
import chess
import chess.pgn
import chess.engine
import chess.syzygy
from multiprocessing import Pool

sys.path.insert(0, "/home/spec/chess-lab/tools")
from degm_pools import band, SF, DEPTH

PGN = os.path.expanduser("~/chess-lab/books/DEGM.pgn")
SYZYGY = os.path.expanduser("~/syzygy")
OUT = "/home/spec/chess-lab/tbpools"
WORKERS = 8
RANK = {"loss": 0, "cursed_loss": 1, "draw": 2,
        "cursed_win": 3, "win": 4}


def sane_board(b):
    return (b.is_valid() and not b.is_game_over()
            and b.king(chess.WHITE) is not None
            and b.king(chess.BLACK) is not None
            and list(b.legal_moves))


def collect():
    """(chapter, fen, book_move_uci, example_idx, ply) per position."""
    out = []
    with open(PGN, encoding="utf-8", errors="replace") as f:
        gi = 0
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            gi += 1
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
            if not sane_board(b):
                continue
            out.append((ch, b.fen(), None, gi, 0))
            ply = 0
            for n in g.mainline():
                try:
                    b.push(n.move)
                except Exception:
                    break
                ply += 1
                if not sane_board(b):
                    if b.is_game_over():
                        break
                    continue
                out.append((ch, b.fen(), None, gi, ply))
    return out


# book moves filled in by main() after collect (needs board replay)


ENG = None
TB = None


def init_worker():
    global ENG, TB
    ENG = chess.engine.SimpleEngine.popen_uci(SF)
    ENG.configure({"Threads": 1, "Hash": 128})
    if os.path.isdir(SYZYGY):
        TB = chess.syzygy.open_tablebase(SYZYGY)


def probe(b):
    """(wdl_from_mover, dtz) or None — mover = b.turn."""
    try:
        dtz = TB.probe_dtz(b)
        wdl = TB.probe_wdl(b)      # from b.turn perspective
        return wdl, dtz
    except Exception:
        return None


def wdl_cat(wdl):
    return {2: "win", 1: "cursed_win", 0: "draw",
            -1: "cursed_loss", -2: "loss"}[wdl]


def grade_one(job):
    ch, fen, book_mv, gi, ply = job
    b = chess.Board(fen)
    mvs = list(b.legal_moves)
    try:
        infos = ENG.analyse(b, chess.engine.Limit(depth=DEPTH),
                            multipv=len(mvs))
    except Exception:
        return None
    children = {}
    cats = {}
    order = []
    for info in infos:
        u = info["pv"][0].uci()
        sc = info["score"].relative
        cat = band(sc)
        sf_band = cat
        dtz = None
        if TB is not None and len(b.piece_map()) <= 5:
            b.push(info["pv"][0])
            pr = probe(b)          # from opponent-to-move perspective
            b.pop()
            if pr is not None:
                wdl, dtz = pr
                cat = wdl_cat(-wdl)             # flip to parent mover
        children[u] = {"cat": cat, "dtz": dtz, "sf_band": sf_band,
                       "sf": sc.score() if not sc.is_mate() else None}
        cats[u] = cat
        order.append(u)

    # AUTHORITY LADDER (operator ruling 2026-09-19): the book's move is
    # TOP-RANKED wherever the book speaks — SF-18 annotates, it never
    # overrides grandmaster analysis. Only the tablebase can disprove a
    # book answer; beyond TB reach the book wins over SF disagreement.
    tb_class = None
    if TB is not None and len(b.piece_map()) <= 5:
        pr = probe(b)
        if pr is not None:
            tb_class = wdl_cat(pr[0])
    book_disproved = False
    if tb_class is not None:
        cat = tb_class
        top = [u for u in cats if RANK[cats[u]] == RANK[cat]]
        if not top:
            # per-move probes missed the TB class (probe-edge cases):
            # fall back to the best band any move actually achieved
            bcat = max(cats.values(), key=lambda x: RANK[x])
            top = [u for u in cats if RANK[cats[u]] == RANK[bcat]]
        if book_mv is not None and book_mv in top:
            best, authority = book_mv, "book+tb"
        else:
            if book_mv is not None:
                book_disproved = True
            best = next((u for u in order if u in top),
                        sorted(top)[0])
            authority = "tb"
    elif book_mv is not None:
        # book authority: the position class is what the book's own
        # move achieves; SF's higher-banded moves do NOT reorder
        cat = cats[book_mv]
        best, authority = book_mv, "book"
    else:
        cat = max(cats.values(), key=lambda x: RANK[x])
        top = [u for u in cats if RANK[cats[u]] == RANK[cat]]
        best = next((u for u in order if u in top), sorted(top)[0])
        authority = "sf"
    return {"fen": fen, "cat": cat, "best": best,
            "children": children, "pool": f"DEGM2_Ch{ch}",
            "pieces": len(b.piece_map()),
            "book_move": book_mv, "book_disproved": book_disproved,
            "authority": authority, "ex": gi, "ply": ply}


def main():
    jobs = collect()
    print(f"collected {len(jobs)} candidate positions", flush=True)

    # attach book moves: replay each example, mark its played move
    by_fen = {}
    with open(PGN, encoding="utf-8", errors="replace") as f:
        gi = 0
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            gi += 1
            fen = g.headers.get("FEN")
            if not fen:
                continue
            try:
                b = chess.Board(fen)
            except Exception:
                continue
            if not sane_board(b):
                continue
            node = g.next()
            if node is not None and node.move in b.legal_moves:
                by_fen[(gi, 0)] = node.move.uci()
            ply = 0
            for n in g.mainline():
                try:
                    b.push(n.move)
                except Exception:
                    break
                ply += 1
                nxt = n.next()
                if nxt is not None and sane_board(b) \
                        and nxt.move in b.legal_moves:
                    by_fen[(gi, ply)] = nxt.move.uci()
    jobs = [(ch, fen, by_fen.get((gi, ply)), gi, ply)
            for ch, fen, _, gi, ply in jobs]
    print(f"book moves attached for "
          f"{sum(1 for j in jobs if j[2])} positions", flush=True)

    per_ch = {}
    n = 0
    with Pool(WORKERS, initializer=init_worker) as p:
        for rec in p.imap(grade_one, jobs, chunksize=8):
            if rec is None:
                continue
            per_ch.setdefault(rec["pool"], []).append(rec)
            n += 1
            if n % 200 == 0:
                print(f"[{n}]", flush=True)
    from collections import Counter
    auth = Counter()
    disproved = 0
    for pool, rows in sorted(per_ch.items()):
        with open(f"{OUT}/{pool}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        a = Counter(r["authority"] for r in rows)
        auth.update(a)
        disproved += sum(1 for r in rows if r["book_disproved"])
        print(f"{pool}: {len(rows)} positions, "
              f"authority={dict(a)}", flush=True)
    json.dump({"pools": {k: len(v) for k, v in per_ch.items()},
               "total": n, "authority": dict(auth),
               "book_disproved": disproved},
              open("/home/spec/chess-lab/degm2_manifest.json", "w"),
              indent=1)
    print(f"DEGM2-REBUILD-COMPLETE: {n} positions, "
          f"authority={dict(auth)}, "
          f"book-disproved={disproved}", flush=True)


if __name__ == "__main__":
    main()
