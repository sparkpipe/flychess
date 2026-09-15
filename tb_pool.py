#!/usr/bin/env python3
"""Build exact-label pools from the lichess Syzygy tablebase API for the
Dvoretsky subset. Each pool entry: (fen, best_uci, category) with category
in {win, loss, draw} from the MOVER's perspective. Probes are cached and
throttled; pools are JSONL so training stages can stream them.

Usage: python3 tb_pool.py KPvK 20000
Material menu: KPvK KQvK KRvK KPvKP KRvKP KRPvKR KBvKP KRvKB KRvKN
"""
import sys, json, time, random, urllib.request, urllib.parse, chess

TB_DIR = "/home/spec/syzygy"
POOL_DIR = "/home/spec/chess-lab/tbpools"
_TB = None


def tb():
    global _TB
    if _TB is None:
        import chess.syzygy
        _TB = chess.syzygy.open_tablebase(TB_DIR)
    return _TB

MATERIAL = {
    "KPvK":  [(chess.PAWN, "w")],
    "KQvK":  [(chess.QUEEN, "w")],
    "KRvK":  [(chess.ROOK, "w")],
    "KBvK":  [(chess.BISHOP, "w")],
    "KNvK":  [(chess.KNIGHT, "w")],
    "KPvKP": [(chess.PAWN, "w"), (chess.PAWN, "b")],
    "KRvKP": [(chess.ROOK, "w"), (chess.PAWN, "b")],
    "KRPvKR": [(chess.ROOK, "w"), (chess.PAWN, "w"), (chess.ROOK, "b")],
    "KBvKP": [(chess.BISHOP, "w"), (chess.PAWN, "b")],
    "KRvKB": [(chess.ROOK, "w"), (chess.BISHOP, "b")],
    "KRvKN": [(chess.ROOK, "w"), (chess.KNIGHT, "b")],
    "KNvKN": [(chess.KNIGHT, "w"), (chess.KNIGHT, "b")],
    "KQvKP": [(chess.QUEEN, "w"), (chess.PAWN, "b")],
    "KQvKR": [(chess.QUEEN, "w"), (chess.ROOK, "b")],
    "KQvKB": [(chess.QUEEN, "w"), (chess.BISHOP, "b")],
    "KQvKN": [(chess.QUEEN, "w"), (chess.KNIGHT, "b")],
    "KBvKB": [(chess.BISHOP, "w"), (chess.BISHOP, "b")],
    "KBvKN": [(chess.BISHOP, "w"), (chess.KNIGHT, "b")],
    "KRvKR": [(chess.ROOK, "w"), (chess.ROOK, "b")],
    "KBBvK": [(chess.BISHOP, "w"), (chess.BISHOP, "w")],
}


def sample_board(name, rng):
    pieces = MATERIAL[name]
    men = 2 + len(pieces)
    while True:
        sqs = rng.sample(range(64), men)
        b = chess.Board(None)
        b.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
        for (pt, col), sq in zip(pieces, sqs[2:]):
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                break
            b.set_piece_at(sq, chess.Piece(pt, chess.WHITE if col == "w"
                                            else chess.BLACK))
        else:
            if abs(chess.square_file(sqs[0]) - chess.square_file(sqs[1])) > 1 \
               or abs(chess.square_rank(sqs[0]) - chess.square_rank(sqs[1])) > 1:
                b.turn = rng.random() < 0.5
                if b.is_valid() and not b.is_game_over():
                    return b


def probe(board):
    """Local Syzygy probe -> same dict shape as the old API result."""
    try:
        wdl = tb().probe_wdl(board)
        dtz = tb().probe_dtz(board)
    except Exception:
        return None
    cat = {2: "win", 0: "draw", -2: "loss"}.get(wdl)
    if cat is None:
        return None
    moves = []
    for mv in board.legal_moves:
        b2 = board.copy(stack=False)
        b2.push(mv)
        try:
            w2 = tb().probe_wdl(b2)
            z2 = tb().probe_dtz(b2)
        except Exception:
            continue
        moves.append({"uci": mv.uci(),
                      "category": {2: "loss", 0: "draw", -2: "win"}.get(w2),
                      "dtz": z2, "dtm": None})
    return {"category": cat, "dtz": dtz, "moves": moves}


def main():
    name = sys.argv[1]
    want = int(sys.argv[2])
    rng = random.Random(hash(name) % 10**6)
    import os
    os.makedirs(POOL_DIR, exist_ok=True)
    path = f"{POOL_DIR}/{name}.jsonl"
    have = 0
    try:
        with open(path) as f:
            have = sum(1 for _ in f)
    except FileNotFoundError:
        pass
    print(f"{name}: pool has {have}, want {want}", flush=True)
    out = open(path, "a")
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor

    def one(_):
        lrng = random.Random(hash(name) ^ threading.get_ident() * 7919)
        while True:
            b = sample_board(name, lrng)
            if b is None:
                continue
            d = probe(b)                     # local syzygy probe
            if d is None or d.get("category") is None:
                time.sleep(0.01)
                continue
            cat = d["category"]
            moves = d.get("moves") or []
            if not moves:
                continue
            opt = {"win": "loss", "loss": "win", "draw": "draw"}[cat]
            cands = [m for m in moves if m.get("category") == opt]
            if not cands:
                continue
            best = min(cands, key=lambda m: m.get("dtz") or 99)
            children = {m["uci"]: {"cat": m.get("category"),
                                   "dtz": m.get("dtz"),
                                   "dtm": m.get("dtm")} for m in moves}
            return json.dumps({"fen": b.fen(), "best": best["uci"],
                               "cat": cat, "dtz": d.get("dtz"),
                               "dtm": d.get("dtm"),
                               "children": children}) + "\n"

    import threading
    with ThreadPoolExecutor(max_workers=16) as ex:
        for line in ex.map(one, range(want - have)):
            out.write(line)
            have += 1
            if have % 1000 == 0:
                out.flush()
                print(f"{name}: {have}/{want} "
                      f"({have/(time.time()-t0):.0f}/s)", flush=True)
    out.close()
    print(f"{name} POOL-DONE {have}", flush=True)


if __name__ == "__main__":
    main()
