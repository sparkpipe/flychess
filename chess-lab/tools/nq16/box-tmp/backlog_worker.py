"""SPARK BACKLOG WORKER — N SF17 processes at depth 12 over a position slice.

Usage (on spark): backlog_worker.py <slice_file> <n_workers> <spark_name>
slice lines: fen<TAB>gid<TAB>ply<TAB>comp
Shard output: ~/backlog_shards/<spark_name>/wK.tsv
  fen<TAB>gid<TAB>ply<TAB>comp<TAB>cp12
Resumable via done-set (board-part key). 5090 collector rsyncs.
"""
import sys, os, time, subprocess, re
import multiprocessing as mp

SF17 = os.environ.get("SF17", os.path.expanduser("~/sf17_arm"))
DEPTH = 12

def worker(args):
    wid, slice_file, n_workers, shard_dir, spark_name = args
    os.makedirs(shard_dir, exist_ok=True)
    p = subprocess.Popen([SF17], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\nsetoption name Threads value 1\nsetoption name Hash value 128\nisready\n")
    p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    local = f"{shard_dir}/w{wid}.tsv"
    done = set()
    if os.path.exists(local):
        for line in open(local):
            done.add(line.split("\t")[0].split(" ")[0])
    f = open(local, "a", buffering=1)
    n, t0, n_all = 0, time.time(), 0
    for i, line in enumerate(open(slice_file)):
        if i % n_workers != wid:
            continue
        n_all += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        fen, gid, ply, comp = parts[0], parts[1], parts[2], parts[3]
        if fen.split(" ")[0] in done:
            continue
        p.stdin.write(f"position fen {fen}\ngo depth {DEPTH}\n")
        p.stdin.flush()
        cp12 = None
        for _ in range(DEPTH * 400):
            ln = p.stdout.readline()
            if ln.startswith("info") and " score " in ln:
                m = re.search(r" depth (\d+) .*?score (cp|mate) (-?\d+)", ln)
                if m and int(m.group(1)) == DEPTH:
                    v = int(m.group(3))
                    if m.group(2) == "mate":
                        v = 30000 + v if v > 0 else -30000 + v
                    cp12 = v
            elif ln.startswith("bestmove"):
                break
        if cp12 is None:
            continue
        f.write(f"{fen}\t{gid}\t{ply}\t{comp}\t{cp12}\n")
        done.add(fen.split(" ")[0])
        n += 1
        if n % 200 == 0:
            r = n / (time.time() - t0)
            print(f"  w{wid}: {n} done, {r:.2f}/s", flush=True)
    p.stdin.write("quit\n")
    p.stdin.flush()
    f.close()
    return wid, n, n_all

if __name__ == "__main__":
    slice_file, n_workers, spark_name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    shard_dir = os.path.expanduser(f"~/backlog_shards/{spark_name}")
    with mp.Pool(n_workers) as pool:
        results = pool.map(worker, [(w, slice_file, n_workers, shard_dir, spark_name)
                                    for w in range(n_workers)])
    for wid, n, n_all in results:
        print(f"w{wid}: {n}/{n_all}")
    print("ALL DONE")
