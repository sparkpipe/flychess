"""SPARK depth-DB worker — runs N SF17 processes over a FEN slice.

Usage (on spark): spark_depth_worker.py <fens_file> <n_workers> <shard_name>
Writes shard TSV (fen<TAB>cp1..cp20) locally; the 5090 collector rsyncs.
Resumable: skips FENs already in the shard.
"""
import sys, os, time, subprocess, re

SF17 = os.path.expanduser("~/sf17_arm")
DEPTH = 20
OUT = os.path.expanduser("~/depth_db_shards")

def worker(wid, fens, shard_path):
    p = subprocess.Popen([SF17], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\nsetoption name Threads value 1\nsetoption name Hash value 128\nisready\n")
    p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    local = shard_path + f".w{wid}"
    done = set()
    for fn in [local] + ([shard_path] if os.path.exists(shard_path) else []):
        if os.path.exists(fn):
            for line in open(fn):
                done.add(line.split("\t")[0].split(" ")[0])
    f = open(local, "a", buffering=1)
    n, t0 = 0, time.time()
    for i, fen in enumerate(fens):
        if i % n_total_slice_jobs != wid_of_slice:
            continue  # not this worker's
        key = fen.split(" ")[0]
        if key in done:
            continue
        p.stdin.write(f"position fen {fen}\ngo depth {DEPTH}\n"); p.stdin.flush()
        cps = [0] * (DEPTH + 1)
        for _ in range(DEPTH * 400):
            line = p.stdout.readline()
            if line.startswith("info") and " score " in line:
                m = re.search(r" depth (\d+) .*?score (cp|mate) (-?\d+)", line)
                if m:
                    d = int(m.group(1)); v = int(m.group(3))
                    if m.group(2) == "mate":
                        v = 30000 + v if v > 0 else -30000 + v
                    if 1 <= d <= DEPTH:
                        cps[d] = v
            elif line.startswith("bestmove"):
                break
        f.write(fen + "\t" + ",".join(str(c) for c in cps[1:]) + "\n")
        n += 1
        if n % 50 == 0:
            print(f"  w{wid}: {n} ({n/(time.time()-t0):.2f}/s)", flush=True)
    p.stdin.write("quit\n"); p.stdin.flush()

if __name__ == "__main__":
    fens_file, n_workers, shard_name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.makedirs(OUT, exist_ok=True)
    shard = f"{OUT}/{shard_name}.tsv"
    fens = [l.strip() for l in open(fens_file) if l.strip()]
    # each worker takes its share of THIS spark's slice
    from concurrent.futures import ThreadPoolExecutor
    def run_wid(wid):
        # subselect for worker wid within the slice
        sub = fens[wid::n_workers]
        p = subprocess.Popen([SF17], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, bufsize=1)
        p.stdin.write("uci\nsetoption name Threads value 1\nsetoption name Hash value 128\nisready\n")
        p.stdin.flush()
        while "readyok" not in p.stdout.readline():
            pass
        local = shard + f".w{wid}"
        done = set()
        for fn in [local] + ([shard] if os.path.exists(shard) else []):
            if os.path.exists(fn):
                for line in open(fn):
                    done.add(line.split("\t")[0].split(" ")[0])
        f = open(local, "a", buffering=1)
        n, t0 = 0, time.time()
        for fen in sub:
            key = fen.split(" ")[0]
            if key in done:
                continue
            p.stdin.write(f"position fen {fen}\ngo depth {DEPTH}\n"); p.stdin.flush()
            cps = [0] * (DEPTH + 1)
            for _ in range(DEPTH * 400):
                line = p.stdout.readline()
                if line.startswith("info") and " score " in line:
                    m = re.search(r" depth (\d+) .*?score (cp|mate) (-?\d+)", line)
                    if m:
                        d = int(m.group(1)); v = int(m.group(3))
                        if m.group(2) == "mate":
                            v = 30000 + v if v > 0 else -30000 + v
                        if 1 <= d <= DEPTH:
                            cps[d] = v
                elif line.startswith("bestmove"):
                    break
            f.write(fen + "\t" + ",".join(str(c) for c in cps[1:]) + "\n")
            n += 1
            if n % 100 == 0:
                print(f"  w{wid}: {n} ({n/(time.time()-t0):.2f}/s)", flush=True)
        p.stdin.write("quit\n"); p.stdin.flush()
        return n
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        total = sum(ex.map(run_wid, range(n_workers)))
    print(f"SPARK_DONE {total}", flush=True)
