"""All-questions singles sweep on DEGM2 (operator directive: solve all
problems in isolation).

Waves of W one-question flies (B=1, shared seed-0 init, shared WT).
Per question: step-0 pass, first-pass step, stable step, and a SPARSE
solution vector (top-K |delta| over concatenated trainable params —
full vectors are 296MB each; top-K=4096 keeps ~all effective support
at 32KB). Resume: questions already in records.jsonl are skipped.

Outputs: singles/records.jsonl, singles/vec_{qi}.npz, singles/summary
"""
import sys
import os
import json
import random
import glob
import time

sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
os.environ.setdefault("ANORM", "1")
os.environ.setdefault("B", "1")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

W = int(os.environ.get("W", "24"))
CAP = int(os.environ.get("CAP", "2000"))
K = int(os.environ.get("K", "4096"))
OUT = "/home/spec/chess-lab/singles"
GATE_EVERY = 10
STABLE_STREAK = 10
LR = float(os.environ.get("LR", "3e-4"))


def load_all():
    rows = []
    pools = sorted(os.path.basename(f)[:-6] for f in glob.glob(
        "/home/spec/chess-lab/tbpools/DEGM2_Ch*.jsonl"))
    for pool in pools:
        rows += fc.load_pools([pool])   # canonical _pre attachment
    return rows


def main():
    flyfeat_cb.feat_vec(chess.Board())
    os.makedirs(OUT, exist_ok=True)
    rows = load_all()
    done = set()
    recf = f"{OUT}/records.jsonl"
    if os.path.exists(recf):
        for line in open(recf):
            done.add(json.loads(line)["fen"])
    todo = [i for i, e in enumerate(rows)
            if e["fen"] not in done]
    # multi-process partitioning: SWEEP_MOD=N SWEEP_REM=k -> this
    # process only trains questions with gi % N == k
    _mod = os.environ.get("SWEEP_MOD")
    if _mod:
        todo = [i for i in todo if i % int(_mod)
                == int(os.environ.get("SWEEP_REM", "0"))]
    print(f"questions total={len(rows)} done={len(done)} "
          f"todo={len(todo)}", flush=True)
    t0 = time.time()
    nw = 0
    base_cat = None                      # seed-0 init params, all waves
    for ws in range(0, len(todo), W):
        wave = todo[ws:ws + W]
        qs = [rows[i] for i in wave]
        models = [tp.build_model(0) for _ in qs]
        for m in models[1:]:
            m.WT = models[0].WT
        if base_cat is None:             # identical init every wave
            base_cat = torch.cat(
                [p.detach().cpu().flatten() for _, p in
                 models[0].named_parameters()])
        opts = [torch.optim.Adam(m.parameters(), lr=LR)
                for m in models]
        rngs = [random.Random(i) for i in wave]
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
            for qi, gi in enumerate(wave):
                e = qs[qi]
                s = st[qi]
                names, flats = [], []
                for n, p in models[qi].named_parameters():
                    names.append(n)
                    flats.append(p.detach().cpu().flatten())
                d = torch.cat(flats) - base_cat      # trained - init
                v, idx = torch.topk(d.abs(), K)
                mx = float(v[0])
                eff = int((d.abs() >= 0.01 * mx).sum())
                np.savez(f"{OUT}/vec_{gi}.npz",
                         idx=idx.numpy().astype(np.int64),
                         val=d[idx].numpy().astype(np.float32),
                         names=np.array(names))
                rec = {"gi": gi, "fen": e["fen"],
                       "pool": e.get("pool"), "cat": e["cat"],
                       "best": e.get("best"),
                       "authority": e.get("authority"),
                       "ply": e.get("ply"),
                       "n_moves": len(list(chess.Board(
                           e["fen"]).legal_moves)),
                       "step0_pass": s["s0"],
                       "first_pass_step": s["first"],
                       "stable_step": s["step"] if s["done"]
                       else None,
                       "final_pair": round(s["pair"], 3),
                       "max_delta": round(mx, 6),
                       "eff_support_1pct": eff}
                f.write(json.dumps(rec) + "\n")
        nw += 1
        el = time.time() - t0
        solved = sum(1 for s in st if s["done"])
        print(json.dumps({"wave": nw, "of": (len(todo) + W - 1) // W,
                          "solved_in_wave": solved, "ofW": len(qs),
                          "elapsed_s": round(el),
                          "eta_h": round(
                              el / nw * ((len(todo) + W - 1) // W - nw)
                              / 3600, 1)}), flush=True)
    print("SWEEP-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
