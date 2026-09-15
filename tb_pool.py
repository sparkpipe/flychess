#!/usr/bin/env python3
"""Build exact-label pools from the lichess Syzygy tablebase API for the
Dvoretsky subset. Each pool entry: (fen, best_uci, category) with category
in {win, loss, draw} from the MOVER's perspective. Probes are cached and
throttled; pools are JSONL so training stages can stream them.

Usage: python3 tb_pool.py KPvK 20000
Material menu: KPvK KQvK KRvK KPvKP KRvKP KRPvKR KBvKP KRvKB KRvKN
"""
import sys, json, time, random, urllib.request, urllib.parse, chess

API = "https://tablebase.lichess.ovh/standard"
POOL_DIR = "/home/spec/chess-lab/tbpools"

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


def probe(fen, tries=3):
    url = API + "?" + urllib.parse.urlencode({"fen": fen})
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return json.load(r)
        except Exception:
            time.sleep(1.0 + a)
    return None


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
    reqs = 0
    t0 = time.time()
    while have < want:
        b = sample_board(name, rng)
        if b is None:
            continue
        d = probe(b.fen())
        reqs += 1
        if d is None or d.get("category") is None:
            continue
        cat = d["category"]                    # mover perspective
        moves = d.get("moves") or []
        if not moves:
            continue
        # best move: optimal child category + min DTZ
        opt = {"win": "loss", "loss": "win", "draw": "draw"}[cat]
        cands = [m for m in moves if m.get("category") == opt]
        if not cands:
            continue
        best = min(cands, key=lambda m: m.get("dtz") or 99)
        # FULL graded move vector: per legal move, the child category + DTZ
        # (both from the mover's perspective) -> graded state-change targets:
        #   no-progress tempo, progress-wasting, the dtz>=100 win-evaporation
        #   cliff, and lost-win/lost-draw catastrophes.
        children = {m["uci"]: {"cat": m.get("category"),
                               "dtz": m.get("dtz"),
                               "dtm": m.get("dtm")} for m in moves}
        out.write(json.dumps({"fen": b.fen(), "best": best["uci"],
                              "cat": cat, "dtz": d.get("dtz"),
                              "dtm": d.get("dtm"),
                              "children": children}) + "\n")
        have += 1
        if have % 500 == 0:
            out.flush()
            print(f"{name}: {have}/{want} "
                  f"({reqs/(time.time()-t0):.0f} req/s)", flush=True)
        time.sleep(0.03)
    out.close()
    print(f"{name} POOL-DONE {have}", flush=True)


if __name__ == "__main__":
    main()
