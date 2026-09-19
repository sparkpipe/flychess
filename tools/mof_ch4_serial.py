"""ch4 size-proof (operator ruling): train ch4's sections SERIALLY to
the 98% exhaustive gate; if a section caps below 98 and is bigger than
MIN_SPLIT, split it in half (feature-space) and continue — the bisection
finds the section size at which a dedicated fly reaches 98%. Everything
serial, one GPU, PASS markers per leaf.
"""
import os
import sys
import json
import random
import numpy as np
import torch
import chess

sys.path.insert(0, "/home/spec/chess-lab")
import fly_curriculum as fc
import flyfeat_cb
from fly_curriculum import DEV, load_pools, FAM_SCORE

CAP = int(os.environ.get("CAP_STEPS", "12000"))
MIN_SPLIT = 8


def train_to_gate(rows, tag, state):
    torch.manual_seed(0)
    rmap = fc.build_retino_map(mode="geo")
    m = fc.FlyCB(len(flyfeat_cb.FEATURE_KEYS), sel_boards=None,
                 readout="variance").to(DEV)
    m.retino = rmap
    m.retino_gain = torch.nn.Parameter(torch.ones(7) * 2.0).to(DEV)
    if os.path.exists(state):
        m.load_state_dict(torch.load(state, weights_only=True),
                          strict=False)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    rng = random.Random(6000 + hash(tag) % 100000)
    FAM_SCORE.update({rows[0].get("pool", tag): 0.0})
    step = 0
    best = 0.0
    while step < CAP:
        for _ in range(500):
            step += 1
            fc.tb_step(m, opt, rows, rng)
        torch.save(m.state_dict(), state + ".tmp")
        os.replace(state + ".tmp", state)
        pair, _, _ = fc.gate_tb(m, rows, random.Random(777),
                                exhaustive=True)
        FAM_SCORE.update({rows[0].get("pool", tag): pair})
        best = max(best, pair)
        print(json.dumps({"tag": tag, "step": step,
                          "exhaustive": round(pair, 4)}), flush=True)
        if pair >= 0.98:
            return True, best
    return False, best


def split_half(rows, tag):
    X = np.stack([flyfeat_cb.feat_vec(chess.Board(e["fen"]))[0]
                  for e in rows]).astype(np.float32)
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
    C = Xn[np.random.default_rng(1).choice(len(Xn), 2, replace=False)]
    for _ in range(25):
        d = ((Xn[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for k in (0, 1):
            m = lab == k
            if m.sum():
                C[k] = Xn[m].mean(0)
    d = ((Xn[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    return [r for i, r in enumerate(rows) if lab[i] == 0], \
        [r for i, r in enumerate(rows) if lab[i] == 1]


def write_pool(rows, pool):
    path = f"/home/spec/chess-lab/tbpools/{pool}.jsonl"
    with open(path, "w") as f:
        for e in rows:
            f.write(json.dumps(dict(e, pool=pool)) + "\n")
    return path


def run(rows, tag, depth=0):
    state = f"/home/spec/chess-lab/flies_mof/l0_{tag}.pt"
    ok, best = train_to_gate(rows, tag, state)
    if ok:
        print(f"PROOF {tag}: PASSED (n={len(rows)})", flush=True)
        return [(tag, len(rows), "PASS")]
    if len(rows) > MIN_SPLIT * 2 and depth < 4:
        a, b = split_half(rows, tag)
        if len(a) < MIN_SPLIT or len(b) < MIN_SPLIT:
            print(f"PROOF {tag}: FAIL (n={len(rows)}, best {best:.3f}, "
                  f"split too small)", flush=True)
            return [(tag, len(rows), f"FAIL@{best:.3f}")]
        print(f"SPLIT {tag} (n={len(rows)}, best {best:.3f}) -> "
              f"{len(a)} + {len(b)}", flush=True)
        pa = run([dict(e, pool=f"{tag}a") for e in a], f"{tag}a",
                 depth + 1)
        pb = run([dict(e, pool=f"{tag}b") for e in b], f"{tag}b",
                 depth + 1)
        return pa + pb
    print(f"PROOF {tag}: FAIL (n={len(rows)}, best {best:.3f})",
          flush=True)
    return [(tag, len(rows), f"FAIL@{best:.3f}")]


def main():
    flyfeat_cb.feat_vec(chess.Board())
    results = []
    for s in ("DEGM_Ch4_s0", "DEGM_Ch4_s1"):
        rows = load_pools([s])
        results += run(rows, s)
    print("CH4-PROOF-COMPLETE " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
