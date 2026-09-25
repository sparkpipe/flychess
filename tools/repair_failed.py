"""SEPARATE repair process (operator directive 2026-09-20): retrain the
sweep's unsolved questions under the fixed cursed-band supervision —
without touching the running sweep or its output.

Reads singles/records.jsonl for stable_step=None questions, trains them
in isolation (W small: fits beside the sweep on the GPU), writes
singles_repair/records.jsonl + vec_{gi}.npz. Idempotent: questions
already repaired are skipped. Re-run anytime; at sweep end re-run to
absorb the failures that accumulated in later waves.

Also verifies each key is self-consistent (best grades top) BEFORE
training; skips-and-logs any row that still fails that check.
"""
import sys
import os
import json
import random
import time

sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
os.environ.setdefault("ANORM", "1")
os.environ.setdefault("B", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

W = int(os.environ.get("W", "3"))
CAP = int(os.environ.get("CAP", "2000"))
K = 4096
IN = "/home/spec/chess-lab/singles"
OUT = "/home/spec/chess-lab/singles_repair"
GATE_EVERY = 10
STABLE_STREAK = 10
LR = float(os.environ.get("LR", "3e-4"))


def main():
    flyfeat_cb.feat_vec(chess.Board())
    os.makedirs(OUT, exist_ok=True)
    pool_cache = {}

    def pool_row(pool, fen):
        if pool not in pool_cache:
            pool_cache[pool] = {r["fen"]: r
                                for r in fc.load_pools([pool])}
        return pool_cache[pool][fen]

    done = set()
    recf = f"{OUT}/records.jsonl"
    if os.path.exists(recf):
        for line in open(recf):
            done.add(json.loads(line)["fen"])

    todo = []
    skipped_key = []
    for line in open(f"{IN}/records.jsonl"):
        r = json.loads(line)
        if r["stable_step"] is not None or r["fen"] in done:
            continue
        e = pool_row(r["pool"], r["fen"])
        v = fc.graded_targets(e, chess.Board(e["fen"]))
        mx = max(v.values())
        if v.get(e["best"]) is None or v[e["best"]] < mx - 1e-9:
            skipped_key.append(r["fen"])
            continue
        todo.append((r, e))
    print(json.dumps({"failed": len(todo) + len(skipped_key),
                      "todo": len(todo),
                      "still_bad_key": len(skipped_key)}), flush=True)

    base_cat = None
    t0 = time.time()
    nw = 0
    for ws in range(0, len(todo), W):
        wave = todo[ws:ws + W]
        qs = [e for _, e in wave]
        models = [tp.build_model(0) for _ in qs]
        for m in models[1:]:
            m.WT = models[0].WT
        if base_cat is None:
            base_cat = torch.cat(
                [p.detach().cpu().flatten() for _, p in
                 models[0].named_parameters()])
        opts = [torch.optim.Adam(m.parameters(), lr=LR)
                for m in models]
        rngs = [random.Random(i) for i in range(len(qs))]
        st = [{"step": 0, "streak": 0, "first": None, "s0": None,
               "done": False, "pair": 0.0} for _ in qs]
        while any(not s["done"] for s in st) and \
                max(s["step"] for s in st) < CAP:
            for qi in range(len(qs)):
                s = st[qi]
                if s["done"]:
                    continue
                if s["step"] % GATE_EVERY == 0:
                    pair, _, _ = fc.gate_tb(
                        models[qi], [qs[qi]], random.Random(777),
                        exhaustive=True)
                    s["pair"] = pair
                    if s["step"] == 0:
                        s["s0"] = bool(pair >= 0.98)
                    s["streak"] = s["streak"] + 1 \
                        if pair >= 0.98 else 0
                    if s["first"] is None and pair >= 0.98:
                        s["first"] = s["step"]
                    if s["streak"] >= STABLE_STREAK:
                        s["done"] = True
                        continue
                fc.tb_step(models[qi], opts[qi], [qs[qi]], rngs[qi])
                s["step"] += 1
        with open(recf, "a") as f:
            for qi, (r, e) in enumerate(wave):
                s = st[qi]
                names, flats = [], []
                for n, p in models[qi].named_parameters():
                    names.append(n)
                    flats.append(p.detach().cpu().flatten())
                d = torch.cat(flats) - base_cat
                v, idx = torch.topk(d.abs(), K)
                np.savez(f"{OUT}/vec_{r['gi']}.npz",
                         idx=idx.numpy().astype(np.int64),
                         val=d[idx].numpy().astype(np.float32),
                         names=np.array(names))
                f.write(json.dumps(
                    {"gi": r["gi"], "fen": r["fen"],
                     "pool": r["pool"], "cat": r["cat"],
                     "best": r["best"],
                     "authority": r.get("authority"),
                     "step0_pass": s["s0"],
                     "first_pass_step": s["first"],
                     "stable_step": s["step"] if s["done"] else None,
                     "final_pair": round(s["pair"], 3),
                     "max_delta": round(float(v[0]), 6),
                     "eff_support_1pct":
                         int((d.abs() >= 0.01 * float(v[0])).sum())})
                    + "\n")
        nw += 1
        print(json.dumps({"repair_wave": nw,
                          "solved": sum(1 for s in st if s["done"]),
                          "ofW": len(qs),
                          "elapsed_s": round(time.time() - t0)}),
              flush=True)
    print("REPAIR-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
