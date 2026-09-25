"""STAGE B — batch static evaluation of the wide-net positions.

Persistent Stockfish workers; each pipelines `position fen` + `eval`
UCI commands in blocks and parses the NNUE static evals. cp is mapped
to win probability for the WINNER of that game: W = 1/(1+exp(-cp/361))
(sign-flipped to winner perspective; mate scores clamp to 0.01/0.99).

Input: gambit/fens.txt  (gid ply winner fen)
Output: gambit/evals.txt (gid ply W)  — same order as input.
"""
import sys
import os
import time
from multiprocessing import Pool

FENS = os.environ.get("FENS_FILE",
                      "/home/spec/chess-lab/gambit/fens.txt")
OUTD = os.environ.get("GAMBIT_DIR", "/home/spec/chess-lab/gambit")
WORKERS = 14
BLOCK = 32
os.makedirs(OUTD, exist_ok=True)


def worker(wid):
    import subprocess
    ranges = []
    size = os.path.getsize(FENS)
    step = size // WORKERS + 1
    for i in range(WORKERS):
        ranges.append((i * step, min((i + 1) * step, size)))
    lo, hi = ranges[wid]
    proc = subprocess.Popen(
        ["/usr/games/stockfish"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1 << 20)
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    while True:                      # wait for uciok
        line = proc.stdout.readline()
        if line.startswith("uciok"):
            break
    proc.stdin.write("setoption name UCI_ShowWDL value true\n"
                     "isready\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if line.startswith("readyok"):
            break
    _sfx = os.environ.get("FENS_SFX", "")
    outp = f"{OUTD}/evals{_sfx}_{wid}.txt"
    t0 = time.time()
    n = 0
    with open(FENS, "rb") as f, open(outp, "w") as w:
        f.seek(lo)
        if lo > 0:
            f.readline()             # discard partial line
        while f.tell() < hi:
            block = []
            while len(block) < BLOCK and f.tell() < hi:
                line = f.readline()
                if not line:
                    break
                parts = line.decode().strip().split(" ", 3)
                if len(parts) < 4:
                    continue
                block.append((parts[0], parts[1], int(parts[2]),
                              parts[3]))
            if not block:
                break
            import math
            for gid, ply, sign, fen in block:
                proc.stdin.write(
                    f"position fen {fen}\nsetoption name UCI_ShowWDL "
                    f"value true\ngo depth 1\n")
            proc.stdin.flush()
            for gid, ply, sign, fen in block:
                mover_w = 0.5
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    if line.startswith("bestmove"):
                        break
                    if " wdl " in line:
                        try:
                            wtk = line.split(" wdl ")[1].split()[:3]
                            wv, dv, lv = (int(x) for x in wtk)
                            mover_w = (wv + dv / 2) / 1000.0
                        except Exception:
                            pass
                wp = mover_w if sign > 0 else 1.0 - mover_w
                w.write(f"{gid} {ply} {wp:.4f}\n")
                n += 1
            if n % 51200 < BLOCK:
                el = time.time() - t0
                print(json.dumps({"wid": wid, "evals": n,
                                  "rate": round(n / max(el, 1))}),
                      flush=True)
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.terminate()
    return {"wid": wid, "evals": n}


import json


def main():
    t0 = time.time()
    with Pool(WORKERS) as p:
        stats = p.map(worker, range(WORKERS))
    tot = sum(s["evals"] for s in stats)
    import glob as _gl
    _all = sorted(_gl.glob(f"{OUTD}/evals{FENS_SFX}_[0-9]*.txt"))
    with open(f"{OUTD}/evals{FENS_SFX}.txt", "w") as out:
        for _f in _all:
            with open(_f) as f:
                while True:
                    b = f.read(1 << 24)
                    if not b:
                        break
                    out.write(b)
    print(json.dumps({"total_evals": tot,
                      "elapsed_s": round(time.time() - t0)}), flush=True)
    print("STAGE-B-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
