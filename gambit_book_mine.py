#!/usr/bin/env python3
"""Mine a gambit opening book from matched lichess game movetext lines.

Input:  matched.pgn — one movetext line per game ending with the result token
        (produced by curl | zstdcat | grep -E -f gambit_prefixes.txt).
Method: each game is anchored at its matching SEED (replayed by SAN), then
        walked on; per (position, move) counts (w/d/b) accumulate to
        MAX_PLIES. Gates: min games by depth + mover score >= SCORE_FLOOR.
Output: gambits.bin (polyglot, weighted) + gambits.epd (readable).
"""
import sys, struct, time, chess, chess.polyglot
import multiprocessing as mp
from collections import defaultdict

MATCHED = "/home/spec/chess-lab/matched.pgn"
OUT = "/home/spec/chess-lab/gambits.bin"
EPD_OUT = "/home/spec/chess-lab/gambits.epd"
MAX_PLIES = 16
SCORE_FLOOR = 0.42
MIN_GAMES = lambda d: 300 if d <= 8 else (120 if d <= 12 else 40)

SEEDS = [
    ("KingsG",      "e2e4 e7e5 f2f4"),
    ("Evans",       "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 b2b4"),
    ("Danish",      "e2e4 e7e5 d2d4 e5d4 c2c3"),
    ("ScotchG",     "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f1c4"),
    ("ItalianG",    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 d2d4"),
    ("TwoKnights",  "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 f3g5"),
    ("TwoKn_d4",    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d4"),
    ("ViennaG",     "e2e4 e7e5 b1c3 g8f6 f2f4"),
    ("Halloween",   "e2e4 e7e5 b1c3 g8f6 c3e5"),
    ("Belgrade",    "e2e4 e7e5 b1c3 g8f6 d2d4 e5d4 c3e5"),
    ("Urusov",      "e2e4 e7e5 f1c4 g8f6 d2d4"),
    ("CenterG",     "e2e4 e7e5 d2d4 e5d4 d1d4"),
    ("Ponziani",    "e2e4 e7e5 g1f3 b8c6 c2c3"),
    ("Morra",       "e2e4 c7c5 d2d4 c5d4 c2c3"),
    ("WingSic",     "e2e4 c7c5 b2b4"),
    ("MilnerBarry", "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3"),
    ("WingFr",      "e2e4 e7e6 b2b4"),
    ("Icelandic",   "e2e4 d7d5 e4d5 g8f6 c2c4"),
    ("NimzoG",      "e2e4 b8c6 d2d4 e7e5"),
    ("Stafford",    "e2e4 e7e5 g1f3 g8f6 f3e5 b8c6"),
    ("Cochrane",    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6"),
    ("Blackmar",    "d2d4 d7d5 e2e4"),
    ("Rousseau",    "e2e4 e7e5 g1f3 f7f6"),
    ("Latvian",     "e2e4 e7e5 g1f3 f7f5"),
    ("Elephant",    "e2e4 e7e5 g1f3 d7d6 d2d4"),
    ("Marshall",    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d5 e4d5 f6d5 f3e5"),
    ("Budapest",    "d2d4 g8f6 c2c4 e7e5"),
    ("Englund",     "d2d4 e7e5"),
    ("Albin",       "d2d4 d7d5 c2c4 e7e5"),
    ("Benko",       "d2d4 g8f6 c2c4 c7c5 d4d5 b7b5"),
    ("Blumenfeld",  "d2d4 g8f6 c2c4 e7e6 g1f3 c7c5 d4d5 b7b5"),
    ("Chigorin",    "d2d4 d7d5 c2c4 b8c6"),
    ("Baltic",      "d2d4 d7d5 c2c4 c8f5"),
]
SEED_MOVES = [(n, l.split()) for n, l in SEEDS]
RESULTS = ("1-0", "0-1", "1/2-1/2", "*")


def tokenize(movetext):
    out = []
    for tok in movetext.split():
        if tok in RESULTS:
            continue
        if tok in ("O-O", "O-O-O", "0-0", "0-0-0"):
            out.append(tok); continue
        stripped_num = tok.lstrip("0123456789")
        if stripped_num.startswith(".") or (tok != stripped_num and stripped_num == ""):
            continue
        t = stripped_num.strip("!?")
        if t:
            out.append(t)
    return out


def walk_game(line):
    line = line.strip()
    if len(line) < 10:
        return None
    if line.endswith("1-0"):
        res = (1, 0, 0)
    elif line.endswith("0-1"):
        res = (0, 0, 1)
    elif "1/2-1/2" in line[-9:]:
        res = (0, 1, 0)
    else:
        return None
    toks = tokenize(line)
    if len(toks) < 3:
        return None
    for name, seed in SEED_MOVES:
        b = chess.Board()
        ok = True
        try:
            for i, uci in enumerate(seed):
                if b.push_san(toks[i]).uci() != uci:
                    ok = False
                    break
        except Exception:
            return None
        if not ok:
            continue
        out = []
        # count (position_before, seed_move) pairs along the seed
        b2 = chess.Board()
        for j, uci in enumerate(seed):
            out.append((b2.epd(), uci, res, j + 1))
            b2.push_uci(uci)
        rest = toks[len(seed):]
        for k, t in enumerate(rest[:MAX_PLIES - len(seed)]):
            try:
                mv = b2.push_san(t)
            except Exception:
                break
            out.append((b2.epd(), mv.uci(), res, len(seed) + k + 1))
            if b2.is_game_over():
                break
        return (name, out)
    return None


def worker(lines):
    agg = defaultdict(lambda: [0, 0, 0])
    for ln in lines:
        r = walk_game(ln)
        if r is None:
            continue
        for epd, uci, (w, d, bl), _ in r:
            a = agg[(epd, uci)]
            a[0] += w; a[1] += d; a[2] += bl
    return dict(agg)


def polyglot_move(board, uci):
    mv = chess.Move.from_uci(uci)
    fr, to = mv.from_square, mv.to_square
    promo = mv.promotion or 0
    pmap = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
    p = pmap.get(promo, 0)
    if board.piece_type_at(fr) == chess.KING and abs(
            (to % 8) - (fr % 8)) > 1:
        to = {chess.G1: chess.H1, chess.C1: chess.A1,
              chess.G8: chess.H8, chess.C8: chess.A8}[to]
    return fr | (to << 6) | (p << 12)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else MATCHED
    lines = open(src, errors="ignore").read().splitlines()
    print(f"matched game lines: {len(lines)}", flush=True)
    pool = mp.Pool(16)
    CH = 20000
    chunks = [lines[i:i + CH] for i in range(0, len(lines), CH)]
    agg = defaultdict(lambda: [0, 0, 0])
    t0 = time.time()
    for i, part in enumerate(pool.imap_unordered(worker, chunks)):
        for k, c in part.items():
            t = agg[k]
            t[0] += c[0]; t[1] += c[1]; t[2] += c[2]
        if (i + 1) % 25 == 0:
            print(f"  chunk {i+1}/{len(chunks)} moves={len(agg)} "
                  f"{time.time()-t0:.0f}s", flush=True)
    pool.close()
    print(f"aggregated {len(agg)} (position,move) pairs "
          f"in {time.time()-t0:.0f}s", flush=True)

    entries = []
    epd_lines = []
    for (epd, uci), (w, d, bl) in agg.items():
        games = w + d + bl
        board = chess.Board()
        try:
            board.set_epd_fen(epd)
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                continue
        except Exception:
            continue
        # depth from epd fullmove field
        parts = epd.split()
        depth = int(parts[-1]) if len(parts) >= 6 else 1
        if games < MIN_GAMES(depth):
            continue
        wscore = (w + 0.5 * d) / games
        mscore = wscore if board.turn == chess.WHITE else 1 - wscore
        if mscore < SCORE_FLOOR:
            continue
        edge = max(0.0, mscore - 0.40)
        weight = min(int(games * edge * edge), 65535)
        if weight == 0:
            continue
        key = chess.polyglot.zobrist_hash(board)
        entries.append((key, polyglot_move(board, uci), weight))
        if games >= 1000:
            epd_lines.append(f"{epd} | {uci} games={games} mscore={mscore:.3f}")
    # dedupe: keep max weight per (key,move)
    best = {}
    for key, pm, wgt in entries:
        k2 = (key, pm)
        if k2 not in best or wgt > best[k2]:
            best[k2] = wgt
    entries = sorted((k, m, w) for (k, m), w in best.items())
    with open(OUT, "wb") as f:
        for key, pm, wgt in entries:
            f.write(struct.pack(">QHHI", key, pm, wgt, 0))
    print(f"wrote {len(entries)} polyglot entries -> {OUT}", flush=True)
    with open(EPD_OUT, "w") as f:
        f.write("\n".join(sorted(epd_lines)) + "\n")
    # readback sanity
    with chess.polyglot.open_reader(OUT) as r:
        n = sum(1 for _ in r)
    print(f"readback OK: {n} entries", flush=True)


if __name__ == "__main__":
    main()
