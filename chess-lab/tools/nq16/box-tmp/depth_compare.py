"""DEPTH COMPARISON EXPERIMENT — 100 random games: d12 vs d20 vs d25.

Samples 100 covered games from the store, evaluates every cp-bearing ply
with SF17.1 `go depth 25` (info stream yields d12, d20, d25 in one run),
histograms per-position deltas and per-game significance.
"""
import sqlite3, subprocess, re, random, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

DB = "/mnt/cold-raid6/chess-audit/wp_fit/store/games.db"
SF = "/usr/games/stockfish"
N_GAMES = 100
random.seed(20261004)

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
gids = [r[0] for r in con.execute(
    "SELECT gid FROM games WHERE covered=1 AND result != '1/2-1/2' ORDER BY RANDOM() LIMIT 60")]
gids += [r[0] for r in con.execute(
    "SELECT gid FROM games WHERE covered=1 AND result = '1/2-1/2' ORDER BY RANDOM() LIMIT 40")]
jobs = []
for gid in gids:
    for (ply, fen, cp12) in con.execute(
            "SELECT ply, fen, cp FROM plies WHERE gid=? AND cp IS NOT NULL ORDER BY ply", (gid,)):
        jobs.append((gid, ply, fen, cp12))
print(f"games: {len(gids)}  positions: {len(jobs):,}", flush=True)

def work(w):
    p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\nsetoption name Threads value 1\nsetoption name Hash value 256\nisready\n")
    p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    out = []
    for i, (gid, ply, fen, cp12) in enumerate(jobs):
        if i % 20 != w:
            continue
        p.stdin.write(f"position fen {fen}\ngo depth 25\n")
        p.stdin.flush()
        cps = {}
        for _ in range(25 * 600):
            line = p.stdout.readline()
            if line.startswith("info") and " score " in line:
                m = re.search(r" depth (\d+) .*?score (cp|mate) (-?\d+)", line)
                if m:
                    d, v = int(m.group(1)), int(m.group(3))
                    if m.group(2) == "mate":
                        v = 30000 + v if v > 0 else -30000 + v
                    cps[d] = v
            elif line.startswith("bestmove"):
                break
        if 12 in cps and 20 in cps and 25 in cps:
            out.append((gid, cps[12], cps[20], cps[25]))
    p.stdin.write("quit\n")
    p.stdin.flush()
    return out

if __name__ != "__main__":
    pass
else:
    import multiprocessing as mp
    mp.set_start_method("fork")
    with mp.Pool(20) as pool:
        results = [r for chunk in pool.map(work, range(20)) for r in chunk]
    print(f"measured: {len(results):,}", flush=True)

    def hist(name, vals):
        b = Counter()
        for v in vals:
            if v < 25: b["<25"] += 1
            elif v < 50: b["25-50"] += 1
            elif v < 100: b["50-100"] += 1
            elif v < 200: b["100-200"] += 1
            elif v < 400: b["200-400"] += 1
            else: b["400+"] += 1
        n = len(vals)
        print(f"\n{name}: n={n:,} mean={sum(vals)/n:.1f}cp")
        for k in ("<25", "25-50", "50-100", "100-200", "200-400", "400+"):
            print(f"  {k:>8} {100*b[k]/n:5.2f}%")

    pairs = {"|d12-d20|": [], "|d20-d25|": [], "|d12-d25|": []}
    pergame = {}
    for gid, c12, c20, c25 in results:
        if abs(c12) > 29000 or abs(c20) > 29000 or abs(c25) > 29000:
            continue
        pairs["|d12-d20|"].append(abs(c12 - c20))
        pairs["|d20-d25|"].append(abs(c20 - c25))
        pairs["|d12-d25|"].append(abs(c12 - c25))
        g = pergame.setdefault(gid, {"a": [], "b": [], "c": []})
        g["a"].append(abs(c12 - c20))
        g["b"].append(abs(c20 - c25))
        g["c"].append(abs(c12 - c25))

    for k, v in pairs.items():
        hist(k, v)

    print(f"\nper-game (n={len(pergame)}): games with >=1 position over threshold:")
    for label, key in (("d12 vs d20", "a"), ("d20 vs d25", "b"), ("d12 vs d25", "c")):
        for thr in (100, 200):
            cnt = sum(1 for g in pergame.values() if max(g[key]) >= thr)
            print(f"  {label} >= {thr}cp: {cnt} games ({cnt}%)")
