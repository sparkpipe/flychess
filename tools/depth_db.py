"""DEPTH-TRAJECTORY DATABASE — strongest engine, fixed depth, all depths 1..20.

Sources: aug bins (29.7M), train_val bins, tb/dvoretsky/c Puzzle FENs (if present).
Per position: cp at every depth 1..20 (one `go depth N` — info stream gives all).
Output: shards of (fen, cp[20]) + done-index. Resumable, idempotent.
Workers: N single-threaded engine processes (Threads=1, deterministic-ish).
"""
import sys, os, glob, time, subprocess, hashlib, random
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np

R = "/mnt/cold-raid6/chess-audit"
OUT = f"{R}/depth_db"
SF17 = "/usr/games/stockfish"
DEPTH = 20

def position_stream():
    """Yield FENs from all sources, deduped by position part."""
    import audit_packer
    seen = set()
    # 1. aug bins (the corpus)
    for b in sorted(glob.glob(f"{R}/expert_bins_both/*.bin")):
        raw = open(b, "rb").read()
        n = len(raw) // 40
        step = max(1, n // 3_000_000)  # cap per-bin to keep first pass broad
        for i in range(0, n, step):
            fen = audit_packer.unpack_sfen(raw[i*40:i*40+32])[0].fen()
            key = fen.split(" ")[0]
            if key not in seen:
                seen.add(key)
                yield fen
    # 2. lichess puzzles if downloaded
    for pz in glob.glob(os.path.expanduser("~/chess-lab/puzzles/*.csv")) + \
              glob.glob(f"{R}/puzzles/*.csv"):
        with open(pz) as f:
            for line in f:
                parts = line.split(",")
                if len(parts) > 1 and "/" in parts[1]:
                    fen = parts[1].strip()
                    key = fen.split(" ")[0]
                    if key not in seen:
                        seen.add(key)
                        yield fen

def worker_main(worker_id, n_workers):
    os.makedirs(f"{OUT}/shards", exist_ok=True)
    p = subprocess.Popen([SF17], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\nsetoption name Threads value 1\nsetoption name Hash value 256\nisready\n")
    p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    shard_f = f"{OUT}/shards/w{worker_id:02d}.tsv"
    done = set()
    if os.path.exists(shard_f):
        for line in open(shard_f):
            done.add(line.split("\t")[0].split(" ")[0])
    f = open(shard_f, "a", buffering=1)
    n_done, t0 = 0, time.time()
    for si, fen in enumerate(position_stream()):
        if si % n_workers != worker_id:
            continue
        key = fen.split(" ")[0]
        if key in done:
            continue
        p.stdin.write(f"position fen {fen}\ngo depth {DEPTH}\n"); p.stdin.flush()
        cps = [None] * (DEPTH + 1)
        best = None
        for _ in range(DEPTH * 400):
            line = p.stdout.readline()
            if line.startswith("info") and " score " in line:
                import re
                m = re.search(r" depth (\d+) .*?score (cp|mate) (-?\d+)", line)
                if m:
                    d = int(m.group(1))
                    v = int(m.group(3))
                    if m.group(2) == "mate":
                        v = 30000 + v if v > 0 else -30000 + v
                    if 1 <= d <= DEPTH:
                        cps[d] = v
            elif line.startswith("bestmove"):
                break
        if cps[DEPTH] is None:
            continue
        # fill forward gaps (rare missing depths)
        last = 0
        for d in range(1, DEPTH + 1):
            if cps[d] is None:
                cps[d] = last
            last = cps[d]
        f.write(fen + "\t" + ",".join(str(c) for c in cps[1:]) + "\n")
        done.add(key)
        n_done += 1
        if n_done % 50 == 0:
            rate = n_done / (time.time() - t0)
            print(f"w{worker_id}: {n_done} done, {rate:.2f}/s", flush=True)
    p.stdin.write("quit\n"); p.stdin.flush()

def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(worker_main, range(n_workers), [n_workers]*n_workers))
    print("DEPTH_DB_PASS_DONE", flush=True)

if __name__ == "__main__":
    main()
