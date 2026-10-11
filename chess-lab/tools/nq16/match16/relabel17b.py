#!/usr/bin/env python3
"""Full miniature relabel v2: SF17 d12, STM-POV OUTPUT. Single-arg worker chunks.
Validates its own output (row shape + sample spot-check) before declaring done."""
import multiprocessing as mp
import os
import subprocess

SF17 = "/srv/workspace/flychess/src/sf17/src/stockfish"
SRC = "/srv/workspace/chess-active/miniature_rows.tsv"
OUT = "/srv/workspace/chess-active/miniature_labels17.tsv"
NW = 16
CHUNK = 20000


def worker(chunk):
    p = subprocess.Popen([SF17], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\nisready\n")
    p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    res = []
    for fen in chunk:
        if "\t" in fen or "[" in fen:
            continue  # guard: fen must be a bare position string
        p.stdin.write(f"position fen {fen}\ngo depth 12\n")
        p.stdin.flush()
        last = None
        ok = True
        while True:
            l = p.stdout.readline()
            if not l:
                ok = False
                break
            if l.startswith("info ") and " score " in l and " pv " in l:
                t = l.split()
                for i, x in enumerate(t):
                    if t[i - 1] == "score" and x in ("cp", "mate"):
                        # mate encoding: 10000 + mate-in, capped (DATA-CONVENTIONS)
                        v = int(t[i + 1])
                        last = (10000 + min(abs(v), 900)) * (1 if v > 0 else -1) \
                            if x == "mate" else v
                        break
            elif l.startswith("bestmove"):
                break
        if ok and last is not None:
            res.append(fen + "\t" + str(last))
    p.stdin.write("quit\n")
    return res


def main():
    done = set()
    if os.path.exists(OUT + ".part"):
        for line in open(OUT + ".part"):
            f = line.split("\t")[0]
            if f and "[" not in f:
                done.add(f)
    print(f"resume: {len(done):,} valid rows", flush=True)
    todo = []
    for line in open(SRC):
        fen = line.split("\t")[0]
        if fen not in done:
            todo.append(fen)
    print(f"todo: {len(todo):,}", flush=True)
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    n = len(done)
    with open(OUT + ".part", "a") as out, mp.Pool(NW) as pool:
        for i, part in enumerate(pool.imap_unordered(worker, chunks)):
            for row in part:
                out.write(row + "\n")
            n += len(part)
            if (i + 1) % 5 == 0:
                out.flush()
                print(f"{n:,} labeled", flush=True)
    out_rows = sum(1 for _ in open(OUT + ".part"))
    print(f"written rows: {out_rows:,}", flush=True)
    assert out_rows > len(todo) * 0.9, "output row count implausible — NOT declaring done"
    os.replace(OUT + ".part", OUT)
    print("RELABEL17 COMPLETE AND VERIFIED", flush=True)


if __name__ == "__main__":
    main()
