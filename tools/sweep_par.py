"""PARALLEL singles sweep — 8 independent fly workers on 8 CUDA
streams (operator directive: real parallelism, not round-robin W).

Each worker owns ONE model/optimizer/stream and pulls questions from a
shared queue; per-question training math is IDENTICAL to the serial
sweep (same seed-0 init, same per-question rng, same protocol) — only
kernel overlap changes. Models are built sequentially at startup so
the shared init is deterministic. Records: same format, written per
question under a lock. Resume by fen as before.
"""
import sys
import os
import json
import random
import queue
import threading
import time

os.environ.setdefault("ANORM", "1")
os.environ.setdefault("B", "1")
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

NW = int(os.environ.get("PAR_W", "8"))
CAP = int(os.environ.get("CAP", "2000"))
K = 4096
OUT = os.environ.get("SWEEP_OUT", "/home/spec/chess-lab/singles")
GATE_EVERY = 10
STABLE_STREAK = 10
LR = 3e-4

rec_lock = threading.Lock()
q_lock = threading.Lock()
q_ctr = [0]
t0 = time.time()
done_ct = [0]


def load_all():
    import glob
    rows = []
    pools = sorted(os.path.basename(f)[:-6] for f in glob.glob(
        "/home/spec/chess-lab/tbpools/DEGM2_Ch*.jsonl"))
    for pool in pools:
        rows += fc.load_pools([pool])
    return rows


def worker(wid, models, opts, rows, todo_q, streams):
    m = models[wid]
    opt = opts[wid]
    stream = streams[wid]
    rng = random.Random(1000 + wid)
    GATE_CH = os.environ.get("GATE_CH", "128")
    os.environ["GATE_CH"] = GATE_CH
    while True:
        with q_lock:
            if not todo_q:
                return
            gi = todo_q.pop()
        e = rows[gi]
        with torch.cuda.stream(stream):
            # identical protocol to sweep_singles
            r = e
            streak = 0
            first = None
            s0 = None
            pair = 0.0
            step = 0
            frng = random.Random(gi)
            opt = torch.optim.Adam(m.parameters(), lr=LR)  # fresh per question, as serial
            while step < CAP:
                p, _, _ = fc.gate_tb(m, [r], random.Random(777),
                                     exhaustive=True)
                pair = p
                if step == 0:
                    s0 = bool(p >= 0.98)
                streak = streak + 1 if p >= 0.98 else 0
                if first is None and p >= 0.98:
                    first = step
                if streak >= STABLE_STREAK:
                    break
                for _ in range(GATE_EVERY):
                    fc.tb_step(m, opt, [r], frng)
                    step += 1
        names, flats = [], []
        with torch.cuda.stream(stream):
            for nm, p in m.named_parameters():
                names.append(nm)
                flats.append(p.detach().cpu().flatten())
        d = torch.cat(flats) - BASE_CAT
        v, idx = torch.topk(d.abs(), K)
        np.savez(f"{OUT}/vec_{gi}.npz",
                 idx=idx.numpy().astype(np.int64),
                 val=d[idx].numpy().astype(np.float32),
                 names=np.array(names))
        rec = {"gi": gi, "fen": e["fen"], "pool": e.get("pool"),
               "cat": e["cat"], "best": e.get("best"),
               "authority": e.get("authority"), "ply": e.get("ply"),
               "n_moves": len(list(chess.Board(e["fen"]).legal_moves)),
               "step0_pass": s0, "first_pass_step": first,
               "stable_step": step if streak >= STABLE_STREAK else None,
               "final_pair": round(pair, 3),
               "max_delta": round(float(v[0]), 6),
               "eff_support_1pct":
                   int((d.abs() >= 0.01 * float(v[0])).sum())}
        with rec_lock:
            with open(f"{OUT}/records.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            done_ct[0] += 1
            if done_ct[0] % 50 == 0:
                el = time.time() - t0
                print(json.dumps({"done": done_ct[0],
                                  "s_per_q": round(el / done_ct[0], 1),
                                  "elapsed_s": round(el)}), flush=True)


BASE_CAT = None


def main():
    global BASE_CAT
    flyfeat_cb.feat_vec(chess.Board())
    os.makedirs(OUT, exist_ok=True)
    rows = load_all()
    done = set()
    if os.path.exists(f"{OUT}/records.jsonl"):
        for line in open(f"{OUT}/records.jsonl"):
            done.add(json.loads(line)["fen"])
    todo = [i for i, e in enumerate(rows) if e["fen"] not in done]
    print(json.dumps({"total": len(rows), "done": len(done),
                      "todo": len(todo), "workers": NW}), flush=True)
    todo_q = todo[::-1]          # LIFO work-stealing
    torch.set_num_threads(4)
    streams = [torch.cuda.Stream() for _ in range(NW)]
    # sequential build: deterministic shared init; each model built
    # INSIDE its own stream so no tensor crosses streams unsynced
    torch.cuda.synchronize()
    models = []
    for w in range(NW):
        with torch.cuda.stream(streams[w]):
            m = tp.build_model(0)
            models.append(m)
    for m in models[1:]:
        m.WT = models[0].WT
    torch.cuda.synchronize()
    BASE_CAT = torch.cat([p.detach().cpu().flatten()
                          for _, p in models[0].named_parameters()])
    opts = [torch.optim.Adam(m.parameters(), lr=LR) for m in models]
    # warm every stream's context
    for s in streams:
        with torch.cuda.stream(s):
            torch.zeros(1, device=fc.DEV)
    threads = [threading.Thread(target=worker,
                                args=(w, models, opts, rows, todo_q,
                                      streams), daemon=True)
               for w in range(NW)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("PAR-SWEEP-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
