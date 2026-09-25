"""GAMBIT POOL GENERATOR: matched gambit games -> training rows.

Game-authority keys (operator doctrine): best = the played winner
move; value ladder from the measured W trajectory (stage B) — the
child's band is the parent-mover band of W after the played move.
cat = band of W at the parent position (winner perspective).

Rows: winner positions from each matched game's trough ply onward
(plies <= 50). Reservoir-sampled to CAP rows. Output:
tbpools/GAMBIT.jsonl (+ precompute via precompute_pools).
"""
import sys
import os
import json
import random
import time

sys.path.insert(0, "/home/spec/chess-lab")
import chess
import chess.pgn

G = "/home/spec/chess-lab/gambit"
OUT = "/home/spec/chess-lab/tbpools/GAMBIT.jsonl"
CAP = int(os.environ.get("CAP", "40000"))
FENS_SFXES = ("", "_hist")


def band(w):
    if w >= 0.85:
        return "win"
    if w >= 0.65:
        return "cursed_win"
    if w >= 0.35:
        return "draw"
    return "cursed_loss"


def main():
    # matched games + troughs
    matched = {}
    for sfx in FENS_SFXES:
        try:
            for gm in json.load(open(f"{G}/games{sfx}.json")):
                matched[gm["gid"]] = gm["trough_ply"]
        except FileNotFoundError:
            pass
    print(json.dumps({"matched_gids": len(matched)}), flush=True)
    # first-position fen -> gid, and gid -> {ply: W}
    firstpos = {}
    Wtraj = {}
    for sfx in FENS_SFXES:
        with open(f"{G}/fens{sfx}.txt") as ff:
            for line in ff:
                parts = line.rstrip("\n").split(" ", 3)
                if len(parts) < 4:
                    continue
                gid, ply, sign, fen = parts
                if gid not in matched:
                    continue
                ply = int(ply)
                if gid not in Wtraj:
                    firstpos[fen.split(" ")[0]] = gid
                    Wtraj[gid] = {}
                Wtraj[gid][ply] = fen
        with open(f"{G}/evals{sfx}.txt") as fe:
            for line in fe:
                gid, ply, w = line.split()
                if gid in matched:
                    Wtraj.setdefault(gid, {})
    # evals are line-aligned with fens but keyed gid,ply — reread both
    for sfx in FENS_SFXES:
        with open(f"{G}/fens{sfx}.txt") as ff, \
                open(f"{G}/evals{sfx}.txt") as fe:
            for fline, eline in zip(ff, fe):
                parts = fline.rstrip("\n").split(" ", 3)
                if len(parts) < 4:
                    continue
                gid, ply = parts[0], int(parts[1])
                if gid not in matched:
                    continue
                Wtraj[gid][ply] = float(eline.split()[2])
    print(json.dumps({"traj_games": len(Wtraj)}), flush=True)

    from multiprocessing import Pool
    size = os.path.getsize(f"{G}/filtered.pgn")
    NW = 6
    step = size // NW + 1
    ranges = [(i * step, min((i + 1) * step, size))
              for i in range(NW)]
    globs = {"firstpos": firstpos, "matched": matched, "Wtraj": Wtraj}
    with Pool(NW) as p:
        parts = p.map(scan_range, [(lo, hi, w, globs)
                                   for w, (lo, hi) in
                                   enumerate(ranges)])
    rows = []
    for part in parts:
        rows += part
    random.Random(5).shuffle(rows)
    rows = rows[:CAP]
    n_games = sum(1 for _ in parts)
    n_hit = 0
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"rows": len(rows)}), flush=True)
    print("GAMBIT-POOL-COMPLETE", flush=True)
    return


def scan_range(args):
    lo, hi, wid, globs = args
    import io
    firstpos = globs["firstpos"]
    matched = globs["matched"]
    Wtraj = globs["Wtraj"]
    rng = random.Random(1100 + wid)
    rows = []
    n_seen = n_games = n_hit = 0
    seen_fens = set()
    t0 = time.time()
    with open(f"{G}/filtered.pgn", "rb") as fb:
        fb.seek(lo)
        if lo > 0:
            while True:
                line = fb.readline()
                if not line or line.startswith(b"[Event "):
                    break
        data = fb.read(max(hi - fb.tell(), 1))
    fh = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8",
                          errors="replace")
    while True:
        g = chess.pgn.read_game(fh)
        if g is None:
            break
        n_games += 1
        if n_games % 500000 == 0:
            print(json.dumps({"scanned": n_games,
                              "hits": n_hit, "rows": len(rows),
                              "s": round(time.time() - t0)}),
                  flush=True)
        res = g.headers.get("Result", "*")
        if res not in ("1-0", "0-1"):
            continue
        board0 = g.board()
        try:
            first_fen = board0.variation_fen if False else None
            node0 = next(iter(g.mainline()))
            b0 = g.board()
            b0.push(node0.move)
            first_key = b0.fen().split(" ")[0]
        except Exception:
            continue
        gid = firstpos.get(first_key)
        if gid is None:
            continue
        n_hit += 1
        winner = chess.WHITE if res == "1-0" else chess.BLACK
        trough = matched[gid]
        traj = Wtraj[gid]
        board = g.board()
        p = 0
        for node in g.mainline():
            p += 1
            if p < trough or p > 50:
                board.push(node.move)
                continue
            fen_before = board.fen()
            if board.turn != winner:
                board.push(node.move)
                continue
            fen_after = fen_before
            bb = chess.Board(fen_before)
            u = node.move.uci()
            w_before = traj.get(p)
            bb.push(node.move)
            w_after = traj.get(p + 1, w_before if w_before
                               is not None else 0.9)
            if w_before is None:
                board.push(node.move)
                continue
            row = {"fen": fen_before, "cat": band(w_before),
                   "best": u,
                   "children": {u: {"cat": band(w_after),
                                    "dtz": None, "sf": None}},
                   "pool": "GAMBIT", "authority": "game",
                   "gid": gid, "ply": p,
                   "W": round(w_before, 4),
                   "pieces": len(bb.piece_map())}
            fk = fen_before.split(" ")[0]
            if fk in seen_fens:
                board.push(node.move)
                continue
            seen_fens.add(fk)
            n_seen += 1
            if len(rows) < CAP:
                rows.append(row)
            else:
                j = rng.randrange(n_seen)
                if j < CAP:
                    rows[j] = row
            board.push(node.move)
    print(json.dumps({"wid": wid, "games": n_games, "hits": n_hit,
                  "rows": len(rows),
                  "s": round(time.time() - t0)}), flush=True)
    return rows


if __name__ == "__main__":
    main()
