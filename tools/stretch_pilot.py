"""STRETCH PILOT — the quick positive PoC (operator): can lambda-mixed
training CREATE the tile radius?

Picks the closest pair of BUILT distbank families (centroid cosine),
trains family A fresh with lambda=0.2 mixed toward its 5 nearest
neighbors, then scores on A's holdout, on neighbor B's border strip
(B's 20 positions nearest A), and on B's full holdout:
  stretched fly  vs  plain A fly (v1 bank)  vs  base model
Success = stretched beats both on B's border while keeping own >=0.95.
"""
import sys
import os
import json
import random

os.environ.setdefault("ANORM", "1")
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

DB = "/home/spec/chess-lab/singles/distbank"
CLOS = "/home/spec/chess-lab/singles/closure"
WIT = "/home/spec/chess-lab/singles/witness"
LAM = 0.2
CAP = 3000
LR = 3e-4
K_NEAR = 5


def main():
    flyfeat_cb.feat_vec(chess.Board())
    built = []
    for line in open(f"{DB}/records.jsonl"):
        r = json.loads(line)
        if r.get("status") == "ok":
            built.append(r["lump"])
    meta = json.load(open(f"{WIT}/meta.json"))
    fens, pools = meta["fens"], meta["pool"]
    row_of = {f: i for i, f in enumerate(fens)}
    state = json.load(open(f"{CLOS}/models_2.json"))

    X = np.load(f"{WIT}/pos_x.npy").astype(np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cents, idxs = {}, {}
    for li in built:
        idx = [row_of[f] for _, f in state[li]["qs"] if f in row_of]
        c = Xn[idx].mean(0)
        cents[li] = c / (np.linalg.norm(c) + 1e-9)
        idxs[li] = idx
    lis = built
    C = np.stack([cents[l] for l in lis])
    S = C @ C.T
    np.fill_diagonal(S, -2)
    a_i, b_i = np.unravel_index(np.argmax(S), S.shape)
    A, B = lis[a_i], lis[b_i]
    print(json.dumps({"familyA": A, "familyB": B,
                      "centroid_cos": round(float(S[a_i, b_i]),
                                            4)}), flush=True)

    pool_cache = {}

    def row_for(pool, fen):
        if pool not in pool_cache:
            pool_cache[pool] = {r["fen"]: r
                                for r in fc.load_pools([pool])}
        return pool_cache[pool][fen]

    def split(li):
        rows = [row_for(pl, f) for pl, f in state[li]["qs"]]
        rng = random.Random(1000 + li)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        cut = max(1, int(0.8 * len(rows)))
        return [rows[i] for i in idx[:cut]], [rows[i]
                                              for i in idx[cut:]]

    trainA, holdA = split(A)
    _, holdB = split(B)
    # B's border strip: 20 of B's TRAIN positions nearest A's centroid
    trB, _ = split(B)
    JB = np.stack([flyfeat_cb.feat_vec_by_fen(r["fen"]) for r in trB])
    JB /= np.linalg.norm(JB, axis=1, keepdims=True) + 1e-9
    border = [trB[k] for k in np.argsort(-(JB @ cents[A]))[:20]]

    # nearest neighbors of A among built, per-pair weights
    order = np.argsort(-C[:, a_i])
    near = [lis[k] for k in order[:K_NEAR] if lis[k] != A]
    cs = np.array([S[a_i, lis.index(n)] for n in near])
    w = np.exp((cs - cs.max()) / 0.10)
    w /= w.sum()
    near_rows = {n: split(n)[0] for n in near}

    model = tp.build_model(0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    trng = random.Random(77)
    step = 0
    while step < CAP:
        for _ in range(25):
            for _ in range(8):
                if trng.random() >= LAM:
                    r = trng.choice(trainA)
                else:
                    n = trng.choices(near, weights=w)[0]
                    r = trng.choice(near_rows[n])
                fc.tb_step(model, opt, [r], trng)
            step += 1
        if step % 500 == 0:
            own, _, _ = fc.gate_tb(model, holdA, random.Random(777),
                                   exhaustive=True)
            bd, _, _ = fc.gate_tb(model, border, random.Random(777),
                                  exhaustive=True)
            print(json.dumps({"step": step, "own": round(own, 3),
                              "borderB": round(bd, 3)}), flush=True)

    def score(m):
        own, _, _ = fc.gate_tb(m, holdA, random.Random(777),
                               exhaustive=True)
        bd, _, _ = fc.gate_tb(m, border, random.Random(777),
                              exhaustive=True)
        hb, _, _ = fc.gate_tb(m, holdB, random.Random(777),
                              exhaustive=True)
        return {"ownA": round(own, 3), "borderB": round(bd, 3),
                "holdB": round(hb, 3)}

    stretched = score(model)
    # plain A fly from the bank
    plain = tp.build_model(0)
    plain_sd = {k: v.detach().cpu().clone()
                for k, v in plain.named_parameters()}
    z = np.load(f"{DB}/vec_{A}.npz")
    with torch.no_grad():
        flat = torch.cat([plain_sd[k].flatten()
                          for k, _ in plain.named_parameters()]
                         ).to(fc.DEV)
        flat.index_add_(0, torch.from_numpy(z["idx"]).to(fc.DEV),
                        torch.from_numpy(z["val"]).to(fc.DEV))
        off = 0
        for _, p in plain.named_parameters():
            n_ = p.numel()
            p.copy_(flat[off:off + n_].view_as(p))
            off += n_
    plainS = score(plain)
    base = tp.build_model(0)
    baseS = score(base)
    out = {"pair": [A, B], "stretched": stretched, "plainA": plainS,
           "base": baseS}
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open("/home/spec/chess-lab/singles/stretch/"
                        "pilot.json", "w"), indent=1)
    print("PILOT-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
