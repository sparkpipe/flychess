"""DISTBANK BUILD: one distribution-fly per closure family (138).

Per family: 80/20 train/held-out split (seeded). Fresh seed-0 fly,
interleaved B=8 batches on the 80% (distribution protocol), 3000 steps,
gates every 50 on BOTH sets. Snapshot at best HELD-OUT gate (the set of
held-out positions passed at that snapshot is recorded — best-snapshot
+ answer-set doctrine: verification, not load-bearing selector).
Singleton families stay point-trained and are marked.

Delta saved sparse top-65536 (distribution edits are denser than
pointers). Output: singles/distbank/vec_{lump}.npz + records.jsonl.
Two workers via DB_MOD/DB_REM (lump index mod partitioning).
"""
import sys
import os
import json
import random
import time

os.environ.setdefault("ANORM", "1")
os.environ.setdefault("B", "8")
os.environ.setdefault("GATE_CH", "2048")
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

CLOS = "/home/spec/chess-lab/singles/closure"
OUT = "/home/spec/chess-lab/singles/distbank"
CAP = int(os.environ.get("CAP", "3000"))
LR = 3e-4
GATE_EVERY = 50
K = 65536
MOD = int(os.environ.get("DB_MOD", "1"))
REM = int(os.environ.get("DB_REM", "0"))
os.makedirs(OUT, exist_ok=True)
RECF = f"{OUT}/records.jsonl"
REC_LOCK_F = f"{OUT}/.lock"


def main():
    flyfeat_cb.feat_vec(chess.Board())
    state = json.load(open(f"{CLOS}/models_2.json"))
    done = set()
    if os.path.exists(RECF):
        for line in open(RECF):
            done.add(json.loads(line)["lump"])
    pool_cache = {}

    def row_for(pool, fen):
        if pool not in pool_cache:
            pool_cache[pool] = {r["fen"]: r
                                for r in fc.load_pools([pool])}
        return pool_cache[pool][fen]

    t0 = time.time()
    n_done = 0
    for li, m in enumerate(state):
        if li % MOD != REM or li in done:
            continue
        qs = [tuple(q) for q in m["qs"]]
        rec = {"lump": li, "size": len(qs)}
        try:
            rows = [row_for(pl, fn) for pl, fn in qs]
        except KeyError as ex:
            rec.update({"status": "missing_row", "err": str(ex)})
            append(rec)
            continue
        if len(rows) < 4:
            # singleton/tiny: point-train all (marked, no holdout)
            train_rows = rows
            hold_rows = []
        else:
            rng = random.Random(1000 + li)
            idx = list(range(len(rows)))
            rng.shuffle(idx)
            cut = max(1, int(0.8 * len(rows)))
            train_rows = [rows[i] for i in idx[:cut]]
            hold_rows = [rows[i] for i in idx[cut:]]

        model = tp.build_model(0)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        trng = random.Random(2000 + li)

        def gate(rrs):
            pair, _, fails = fc.gate_tb(model, rrs, random.Random(777),
                                        exhaustive=True)
            return pair, fails

        best = {"hold": -1.0, "train": -1.0, "step": 0, "sd": None,
                "hold_pass": None}
        step = 0
        while step < CAP:
            for _ in range(GATE_EVERY):
                fc.tb_step(model, opt, train_rows, trng)
                step += 1
            tr_g, _ = gate(train_rows)
            ho_g, ho_f = gate(hold_rows) if hold_rows else (1.0, {})
            if ho_g > best["hold"]:
                best.update({"hold": ho_g, "train": tr_g, "step": step,
                             "sd": {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict()
                                    .items()},
                             "hold_pass": ho_g})
        if best["sd"] is not None:
            model.load_state_dict(best["sd"])
        base = tp.build_model(0)
        base_sd = {k: v.detach().cpu().clone()
                   for k, v in base.named_parameters()}
        names, flats = [], []
        for n_, p in model.named_parameters():
            names.append(n_)
            flats.append(p.detach().cpu().flatten())
        d = torch.cat(flats) - torch.cat(
            [base_sd[n_].flatten() for n_ in names])
        v, idxk = torch.topk(d.abs(), min(K, d.numel()))
        np.savez(f"{OUT}/vec_{li}.npz",
                 idx=idxk.numpy().astype(np.int64),
                 val=d[idxk].numpy().astype(np.float32),
                 names=np.array(names))
        rec.update({
            "status": "ok", "n_train": len(train_rows),
            "n_hold": len(hold_rows),
            "train_at_best": round(best["train"], 3),
            "hold_at_best": round(best["hold"], 3),
            "best_step": best["step"],
            "elapsed_s": round(time.time() - t0)})
        append(rec)
        n_done += 1
        print(json.dumps(rec), flush=True)
    print(json.dumps({"WORKER-DONE": True, "REM": REM,
                      "built": n_done}), flush=True)


def append(rec):
    import fcntl
    with open(RECF, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
