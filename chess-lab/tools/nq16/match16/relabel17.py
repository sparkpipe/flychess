#!/usr/bin/env python3
"""Full miniature relabel: SF17 d12, STM-POV OUTPUT (bin convention — NO white flip).
Resumable; writes chess-active/miniature_labels17.tsv. ~43.5M positions."""
import multiprocessing as mp
import os

SF17 = "/srv/workspace/flychess/src/sf17/src/stockfish"
SRC = "/srv/workspace/chess-active/miniature_rows.tsv"
OUT = "/srv/workspace/chess-active/miniature_labels17.tsv"
NW = 16


def worker(args):
    chunk, = args,
    import subprocess
    p = subprocess.Popen([SF17], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\nisready\n")
    p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    res = []
    for fen in chunk:
        p.stdin.write(f"position fen {fen}\ngo depth 12\n")
        p.stdin.flush()
        last = None
        ok = True
        while True:
            l = p.stdout.readline()
            if not l:
                ok = False
                break
            if l.startswith("info ") and " score cp " in l and " pv " in l:
                t = l.split()
                for i, x in enumerate(t):
                    if x == "cp":
                        last = int(t[i + 1])
            elif l.startswith("bestmove"):
                break
        if ok and last is not None:
            res.append(f"{fen}\t{last}")
    p.stdin.write("quit\n")
    return res


def main():
    # single append-only progress file; OUT only appears on full completion
    if os.path.exists(OUT) and not os.path.exists(OUT + ".part"):
        import shutil
        shutil.copy(OUT, OUT + ".part")
    done = set()
    if os.path.exists(OUT + ".part"):
        for line in open(OUT + ".part"):
            done.add(line.split("\t")[0])
    print(f"resume: {len(done):,} already labeled", flush=True)
    todo = []
    for line in open(SRC):
        fen = line.split("\t")[0]
        if fen not in done:
            todo.append(fen)
    print(f"todo: {len(todo):,}", flush=True)
    CHUNK = 20000
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    with open(OUT + ".part", "a") as out:
        with mp.Pool(NW) as pool:
            for i, part in enumerate(pool.imap_unordered(worker, [(c,) for c in chunks])):
                for row in part:
                    out.write(row + "\n")
                if (i + 1) % 5 == 0:
                    out.flush()
                    print(f"{(i + 1) * CHUNK:,} done of {len(todo):,}", flush=True)
    os.replace(OUT + ".part", OUT)
    print("RELABEL17 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
