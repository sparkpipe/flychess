"""STAGE A — wide-net FEN dump for initiative-trajectory mining.

Input: the full OTB archive (Lumbras complete PGN).
Pass 1: byte-level prefilter — keep only decisive games (result 1-0/0-1)
  (fast text scan; python-chess never sees rejected games).
Pass 2: 8-way parallel parse of the filtered games; per game keep
  winner Elo >= 2400 & loser >= 2000 (or no Elo headers: archive-era),
  and emit every position of plies 1..32 as:
    gid ply winner fen
Output: gambit/fens.txt (plus filtered.pgn, stats).
"""
import sys
import os
import json
import glob
import time
from multiprocessing import Pool

PGN = "/home/spec/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn"
OUT = "/home/spec/chess-lab/gambit"
FENS_SFX = os.environ.get("FENS_SFX", "")
WORKERS = 6
MAX_PLIES = 50          # 25 full moves
os.makedirs(OUT, exist_ok=True)


def prefilter():
    """byte scan: keep decisive-game blocks."""
    src = f"{OUT}/filtered.pgn"
    if os.path.exists(src) and os.path.getsize(src) > 0:
        print("prefilter cached", flush=True)
        return os.path.getsize(src)
    t0 = time.time()
    n_in = n_out = 0
    with open(PGN, "rb") as f, open(src + ".tmp", "wb") as w:
        buf = []
        for line in f:
            if line.startswith(b"[Event ") and buf:
                block = b"".join(buf)
                buf = []
                n_in += 1
                if b'[Result "1-0"]' in block or b'[Result "0-1"]' in block:
                    w.write(block)
                    n_out += 1
            buf.append(line)
        tail = b"".join(buf)
        if b'[Result "1-0"]' in tail or b'[Result "0-1"]' in tail:
            w.write(tail)
            n_out += 1
    os.rename(src + ".tmp", src)
    print(json.dumps({"prefilter_s": round(time.time() - t0),
                      "games_in": n_in, "games_decisive": n_out}),
          flush=True)
    return os.path.getsize(src)


def ranges(size, n):
    out = []
    step = size // n + 1
    for i in range(n):
        out.append((i * step, min((i + 1) * step, size)))
    return out


def parse_range(args):
    lo, hi, wid = args
    import chess
    import chess.pgn
    import io
    outp = f"{OUT}/fens{FENS_SFX}_{wid}.txt"
    n_games = n_fens = 0
    with open(f"{OUT}/filtered.pgn", "rb") as f:
        f.seek(lo)
        if lo > 0:
            # snap forward to the next game boundary
            while True:
                line = f.readline()
                if not line or line.startswith(b"[Event "):
                    break
        data = f.read(max(hi - f.tell(), 1))
    text = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8",
                            errors="replace")
    with open(outp, "wb") as w:
        while True:
            g = chess.pgn.read_game(text)
            if g is None:
                break
            try:
                res = g.headers.get("Result", "*")
                if res not in ("1-0", "0-1"):
                    continue
                winner = chess.WHITE if res == "1-0" else chess.BLACK
                wel = int(g.headers.get("WhiteElo", 0) or 0)
                bel = int(g.headers.get("BlackElo", 0) or 0)
                we, le = (wel, bel) if winner == chess.WHITE \
                    else (bel, wel)
                elo_given = not (wel == 0 and bel == 0)
                # operator ruling: archive games are assumed good;
                # pre-1930 games qualify regardless of (back-filled)
                # Elo — quality is enforced by the gambit shape itself
                date = g.headers.get("Date", "") or ""
                yr = 0
                try:
                    yr = int(date[:4])
                except Exception:
                    pass
                hist = 0 < yr <= int(
                    os.environ.get("HIST_YEAR", "1930"))
                if os.environ.get("HIST_ONLY") == "1" and not hist:
                    continue
                if not hist and elo_given \
                        and (we < 2400 or le < 2000):
                    continue
            except Exception:
                continue
            gid = f"{wid}_{n_games}"
            n_games += 1
            b = g.board()
            sign = 1 if winner == chess.WHITE else -1
            ply = 0
            for node in g.mainline():
                ply += 1
                if ply > MAX_PLIES:
                    break
                b.push(node.move)
                w.write((f"{gid} {ply} {sign} {b.fen()}\n")
                        .encode())
                n_fens += 1
    return {"wid": wid, "games": n_games, "fens": n_fens}


def main():
    size = prefilter()
    t0 = time.time()
    with Pool(WORKERS) as p:
        stats = p.map(parse_range, [(lo, hi, i)
                                    for i, (lo, hi) in
                                    enumerate(ranges(size, WORKERS))])
    tot_g = sum(s["games"] for s in stats)
    tot_f = sum(s["fens"] for s in stats)
    import glob as _gl
    _all = sorted(_gl.glob(f"{OUT}/fens{FENS_SFX}_[0-9]*.txt"))
    with open(f"{OUT}/fens{FENS_SFX}.txt", "wb") as out:
        for _f in _all:
            with open(_f, "rb") as f:
                while True:
                    b = f.read(1 << 24)
                    if not b:
                        break
                    out.write(b)
    print(json.dumps({"games": tot_g, "fens": tot_f,
                      "elapsed_s": round(time.time() - t0)}), flush=True)
    print("STAGE-A-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
