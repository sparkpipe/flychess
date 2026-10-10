"""GAMBIT POOL v2 — exact gid reproduction (the ply-1 fen join was
opening-ambiguous; gids are rebuilt by replaying stage A's exact
partitioning + filters, so attribution is correct by construction).

Two passes, exactly as run historically:
  M (modern, fens.txt): WORKERS=6 ranges over filtered.pgn; filter =
    decisive + (no-Elo OR winner>=2400 & loser>=2000)   [pre-date-patch]
  H (hist, fens_hist.txt): same ranges; filter = yr<=1930 only
gid = f"{wid}_{n}" with n incremented ONLY for games passing the
pass filter — identical to stage A. Then: matched gids (from
games.json / games_hist.json, pass-tagged) emit winner-move rows,
game-authority, W-trajectory values.
"""
import sys
import os
import json
import random
import time
import io
from multiprocessing import Pool

sys.path.insert(0, "/home/spec/chess-lab")
import chess
import chess.pgn

G = "/home/spec/chess-lab/gambit"
OUT = "/home/spec/chess-lab/tbpools/GAMBIT.jsonl"
CAP = int(os.environ.get("CAP", "40000"))
NW = 6


def band(w):
    if w >= 0.85:
        return "win"
    if w >= 0.65:
        return "cursed_win"
    if w >= 0.35:
        return "draw"
    return "cursed_loss"


def load_matched():
    M = {}
    for gm in json.load(open(f"{G}/games.json")):
        M[("M", gm["gid"])] = gm["trough_ply"]
    for gm in json.load(open(f"{G}/games_hist.json")):
        M[("H", gm["gid"])] = gm["trough_ply"]
    Wt = {}
    for tag, sfx in (("M", ""), ("H", "_hist")):
        with open(f"{G}/fens{sfx}.txt") as ff, \
                open(f"{G}/evals{sfx}.txt") as fe:
            for fline, eline in zip(ff, fe):
                parts = fline.rstrip("\n").split(" ", 3)
                if len(parts) < 4:
                    continue
                gid, ply = parts[0], int(parts[1])
                key = (tag, gid)
                if key not in M:
                    continue
                Wt.setdefault(key, {})[ply] = float(eline.split()[2])
    return M, Wt


GLOBS = None


def init_worker(m, w):
    global GLOBS
    GLOBS = (m, w)


def scan(args):
    lo, hi, wid = args
    matched, Wt = GLOBS
    import io as _io
    rng = random.Random(1100 + wid)
    rows = []
    n_seen = 0
    seen = set()
    t0 = time.time()
    size = os.path.getsize(f"{G}/filtered.pgn")
    step = size // NW + 1
    with open(f"{G}/filtered.pgn", "rb") as fb:
        fb.seek(lo)
        if lo > 0:
            while True:
                line = fb.readline()
                if not line or line.startswith(b"[Event "):
                    break
        data = fb.read(max(hi - fb.tell(), 1))
    fh = _io.TextIOWrapper(_io.BytesIO(data), encoding="utf-8",
                           errors="replace")
    cntM = 0
    cntH = 0
    while True:
        g = chess.pgn.read_game(fh)
        if g is None:
            break
        res = g.headers.get("Result", "*")
        if res not in ("1-0", "0-1"):
            continue
        try:
            wel = int(g.headers.get("WhiteElo", 0) or 0)
            bel = int(g.headers.get("BlackElo", 0) or 0)
        except Exception:
            continue
        we, le = (wel, bel) if res == "1-0" else (bel, wel)
        elo_given = not (wel == 0 and bel == 0)
        date = g.headers.get("Date", "") or ""
        yr = 0
        try:
            yr = int(date[:4])
        except Exception:
            pass
        hist = 0 < yr <= 1930
        # exact stage-A acceptances
        accept_M = (not elo_given) or (we >= 2400 and le >= 2000)
        accept_H = hist
        gid_m = f"{wid}_{cntM}"
        if accept_M:
            cntM += 1
        gid_h = f"{wid}_{cntH}"
        if accept_H:
            cntH += 1
        winner_side = chess.WHITE if res == "1-0" else chess.BLACK
        for tag, gid, ok in (("M", gid_m, accept_M),
                             ("H", gid_h, accept_H)):
            if not ok:
                continue
            key = (tag, gid)
            if key not in matched:
                continue
            trough = matched[key]
            traj = Wt.get(key, {})
            board = g.board()
            p = 0
            for node in g.mainline():
                p += 1
                fen_before = board.fen()
                u = node.move.uci()
                if trough <= p <= 50 \
                        and board.turn == winner_side:
                    w_b = traj.get(p)
                    w_a = traj.get(p + 1, w_b)
                    if w_b is not None:
                        fk = fen_before.split(" ")[0]
                        if fk not in seen:
                            seen.add(fk)
                            n_seen += 1
                            row = {"fen": fen_before,
                                   "cat": band(w_b), "best": u,
                                   "children": {u: {"cat": band(w_a),
                                                    "dtz": None,
                                                    "sf": None}},
                                   "pool": "GAMBIT",
                                   "authority": "game",
                                   "gid": gid, "pass": tag, "ply": p,
                                   "W": round(w_b, 4),
                                   "pieces": len(board.piece_map())}
                            if len(rows) < CAP:
                                rows.append(row)
                            else:
                                j = rng.randrange(n_seen)
                                if j < CAP:
                                    rows[j] = row
                board.push(node.move)
    print(json.dumps({"wid": wid, "rows": len(rows),
                      "s": round(time.time() - t0)}), flush=True)
    return rows


def main():
    matched, Wt = load_matched()
    print(json.dumps({"matched": len(matched),
                      "traj": len(Wt)}), flush=True)
    size = os.path.getsize(f"{G}/filtered.pgn")
    step = size // NW + 1
    ranges = [(i * step, min((i + 1) * step, size))
              for i in range(NW)]
    with Pool(NW, initializer=init_worker,
              initargs=(matched, Wt)) as p:
        parts = p.map(scan, [(lo, hi, w)
                             for w, (lo, hi) in enumerate(ranges)])
    rows = [r for part in parts for r in part]
    random.Random(5).shuffle(rows)
    rows = rows[:CAP]
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(json.dumps({"rows": len(rows),
                      "by_pass": dict(Counter(r["pass"]
                                               for r in rows)),
                      "by_cat": dict(Counter(r["cat"]
                                             for r in rows))}),
          flush=True)
    print("GAMBIT-POOL-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
