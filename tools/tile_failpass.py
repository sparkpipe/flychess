"""FAIL -> PASS demo (operator): the tiling mechanism on ONE position.

1. FIND: among built neighboring families (medium-distance pair),
   locate a position lying BETWEEN them (near-equal feature cosine to
   both centroids) where BOTH flies and the BASE model fail (<0.98).
2. STRETCH: retrain fly A mixed with B's train rows (lambda=0.2) and
   fly B mixed with A's rows — mutual pull, exactly toward each other.
3. VERIFY: the gap position on both stretched flies; own holdouts
   checked for regression.

Prints the position, before/after gates. Output:
singles/stretch/failpass.json
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
CAP = int(os.environ.get("CAP", "2000"))
LR = 3e-4


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
    fam_of = {}
    for li, m in enumerate(state):
        for pl, f in m["qs"]:
            if f in row_of:
                fam_of[row_of[f]] = li

    X = np.load(f"{WIT}/pos_x.npy").astype(np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cents = {}
    for li in built:
        idx = [row_of[f] for _, f in state[li]["qs"] if f in row_of]
        c = Xn[idx].mean(0)
        cents[li] = c / (np.linalg.norm(c) + 1e-9)
    lis = built
    C = np.stack([cents[l] for l in lis])
    S = C @ C.T
    np.fill_diagonal(S, -2)

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

    base = tp.build_model(0)
    base_sd = {k: v.detach().cpu().clone()
               for k, v in base.named_parameters()}
    base_cat = torch.cat([base_sd[k].flatten()
                          for k, _ in base.named_parameters()])

    def apply_bank_fly(li):
        z = np.load(f"{DB}/vec_{li}.npz")
        with torch.no_grad():
            for k, p in base.named_parameters():
                p.copy_(base_sd[k])
            flat = base_cat.to(fc.DEV)
            flat.index_add_(0, torch.from_numpy(z["idx"]).to(fc.DEV),
                            torch.from_numpy(z["val"]).to(fc.DEV))
            off = 0
            for _, p in base.named_parameters():
                n_ = p.numel()
                p.copy_(flat[off:off + n_].view_as(p))
                off += n_

    def g(m, rows):
        p, _, _ = fc.gate_tb(m, rows, random.Random(777),
                             exhaustive=True)
        return p

    # neighbor pairs, medium distance (0.7..0.95 cosine)
    pairs = []
    for i in range(len(lis)):
        for j in range(i + 1, len(lis)):
            c = float(S[i, j])
            if 0.70 <= c <= 0.95:
                pairs.append((c, lis[i], lis[j]))
    pairs.sort(reverse=True)
    print(json.dumps({"candidate_pairs": len(pairs)}), flush=True)

    rng = random.Random(1)
    demo = None
    tested = 0
    for c, A, B in pairs[:12]:
        ai, bi = lis.index(A), lis.index(B)
        # in-between candidates: exclude A and B members
        ca = Xn @ C[ai]
        cb = Xn @ C[bi]
        mid = np.minimum(ca, cb)
        balanced = np.abs(ca - cb) < 0.05
        cand = [i for i in range(len(fens))
                if balanced[i] and mid[i] > 0.35
                and fam_of.get(i) not in (A, B)]
        rng.shuffle(cand)
        trainA, holdA = split(A)
        trainB, holdB = split(B)
        for pi in cand[:12]:
            row = row_for(pools[pi], fens[pi])
            tested += 1
            apply_bank_fly(A)
            fa = g(base, [row])
            apply_bank_fly(B)
            fb = g(base, [row])
            with torch.no_grad():
                for k, p in base.named_parameters():
                    p.copy_(base_sd[k])
            fbase = g(base, [row])
            if fa < 0.98 and fb < 0.98 and fbase < 0.98:
                demo = {"A": A, "B": B, "cos": round(c, 4),
                        "pos": pi, "fen": fens[pi],
                        "flyA": round(fa, 3), "flyB": round(fb, 3),
                        "base": round(fbase, 3)}
                break
        if demo:
            break
    print(json.dumps({"searched": tested, "demo": demo}), flush=True)
    if not demo:
        print("NO-GAP-FOUND", flush=True)
        return
    A, B = demo["A"], demo["B"]
    trainA, holdA = split(A)
    trainB, holdB = split(B)
    demo_row = row_for(pools[demo["pos"]], demo["fen"])

    def stretch(train_self, train_other):
        m = tp.build_model(0)
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        tr = random.Random(99)
        for step in range(CAP):
            for _ in range(8):
                r = tr.choice(train_self) if tr.random() >= LAM \
                    else tr.choice(train_other)
                fc.tb_step(m, opt, [r], tr)
        return m

    mA = stretch(trainA, trainB)
    mB = stretch(trainB, trainA)
    out = {**demo,
           "stretched_flyA_on_gap": round(g(mA, [demo_row]), 3),
           "stretched_flyB_on_gap": round(g(mB, [demo_row]), 3),
           "stretched_flyA_ownhold": round(g(mA, holdA), 3),
           "stretched_flyB_ownhold": round(g(mB, holdB), 3),
           "plain_flyA_ownhold": round((apply_bank_fly(A),
                                        g(base, holdA))[1], 3),
           "plain_flyB_ownhold": round((apply_bank_fly(B),
                                        g(base, holdB))[1], 3)}
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open("/home/spec/chess-lab/singles/stretch/"
                        "failpass.json", "w"), indent=1)
    print("FAILPASS-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
