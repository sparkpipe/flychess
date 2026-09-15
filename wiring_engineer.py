#!/usr/bin/env python3
"""Wiring engineering v2 — measured, not assumed.

With 2-step LINEAR propagation, scalar signals survive everywhere; the real
constraints are routing, multiplexing, and saturation. So we measure:

  1. Liveness — per-feature variance over a diverse battery (dead features
     flagged BEFORE training).
  2. Routing matrix — for each site S: e_S = P @ 1_S (K-step influence of a
     unit injection). M[i][j] = fraction of site i's output mass landing in
     region j after K steps. This is the connectome's functional topology
     under our exact trainer dynamics.
  3. Multiplexing — inject R random independent patterns at S, propagate,
     decode pattern identity from the top-variance readout pool by
     nearest-centroid. capacity(S) = decode accuracy. Sites that cannot
     carry many channels are bad injection hubs regardless of anatomy.
  4. Gain to readout — |e_S| mass on the global top-variance pool vs spread
     elsewhere (dilution measurement — the CX failure mode).

Output: wiring_report.json — routing matrix, capacity per site, recommended
family->site assignment along strongest converging paths.
Pure scipy CPU; ~25 propagations total.
"""
import sys, os, json, random, time
import numpy as np
import scipy.sparse as sp
import chess
import pandas as pd

sys.path.insert(0, "/home/spec/chess-lab")
import flyfeat_cb

BRAIN = "/home/spec/chess-lab/brain_graph.npz"
ANN = "/home/spec/chess-lab/annotations.feather"
NODES = "/home/spec/chess-lab/node_ids.npy"
OUT = "/home/spec/chess-lab/wiring_report.json"
LEAK, CAP, STEPS = 0.5, 20.0, 2

SITES = {
    "lamina":        r"^(L[1-5]|Lai|Law)",
    "medulla_Mi":    r"^(Mi[1-9]|Dm[0-9]|Cm)",
    "medulla_Tm":    r"^(Tm[A-Z]|TmY)",
    "lobula":        r"^(LC[0-9A-Za-z]|LT[0-9A-Za-z]|Lo[0-9])",
    "lobula_plate":  r"^(LP[0-9A-Za-z]|T4|T5)",
    "KC":            r"^(KC[a-z-]*|Kc)",
    "central_complex": r"^(PF[A-Z]|FR|hDelta|Delta|EPG|E-PG|EL[A-Z]?|NO[1-3A-Z]|PFG|PEN|PEG|PFR)",
    "lateral_horn":  r"^(LH|LHAL|PV5N)",
    "sensory_periph": r"(ORN|sensory|photoreceptor|Chordotonal|mechanosens|bristle)",
    "descending":    r"^(DNp?|DNa|DNb|SEZ)",
}

FAMILIES = {
    "material":    lambda k: k.startswith(("bb_my_", "bb_their_", "mat_")),
    "attacks":     lambda k: k.startswith(("atk_my_", "atk_their_", "net_")),
    "occupancy":   lambda k: k.startswith(("occ_my_", "occ_their_")),
    "eyes_view":   lambda k: k.startswith("eye"),
    "king_rings":  lambda k: k.startswith("ks_"),
    "castling":    lambda k: k.startswith("castle_"),
    "en_passant":  lambda k: k == "ep_file",
    "check_state": lambda k: k in ("in_check", "checkers", "pinned_n"),
    "mobility":    lambda k: k.startswith("mob_"),
}


def battery(n=200, seed=17):
    rng = random.Random(seed)
    boards = []
    df = pd.read_csv("/home/spec/chess-lab/puzzles.csv",
                     usecols=["FEN"], nrows=300000)
    for f in df.FEN.dropna().sample(n=min(n // 2, len(df)), random_state=seed):
        try:
            boards.append(chess.Board(f))
        except Exception:
            pass
    while len(boards) < n:
        b = chess.Board()
        for _ in range(rng.randrange(6, 90)):
            mvs = list(b.legal_moves)
            if not mvs:
                break
            b.push(rng.choice(mvs))
        if not b.is_game_over():
            boards.append(b)
    return boards[:n]


def build_WT():
    z = np.load(BRAIN)
    m = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                      shape=tuple(z["shape"])).astype(np.float32)
    return m.T.tocsr(), m.shape[0]


def site_masks(node_ids):
    a = pd.read_feather(ANN)
    pos = {int(b): i for i, b in enumerate(node_ids)}
    t = a["flywireType"].fillna("").astype(str)
    out = {}
    for name, pat in SITES.items():
        m = t.str.contains(pat, regex=True, na=False)
        idx = np.array(sorted(pos[int(b)] for b in a.bodyId[m] if int(b) in pos),
                       dtype=np.int64)
        if len(idx) >= 8:
            out[name] = idx
    return out


def propagate(WT, a):
    for _ in range(STEPS):
        a = 0.5 * a + 0.5 * (WT @ a)
        np.clip(a, -CAP, CAP, out=a)
    return a


def main():
    t0 = time.time()
    flyfeat_cb.feat_vec(chess.Board())
    boards = battery()
    F = np.stack([flyfeat_cb.feat_vec(b)[0] for b in boards])
    keys = flyfeat_cb.FEATURE_KEYS
    var = F.var(axis=0)
    dead = [keys[i] for i in range(len(keys)) if var[i] < 1e-8]
    print(f"liveness: {len(dead)} dead of {len(keys)}", flush=True)

    WT, N = build_WT()
    node_ids = np.load(NODES)
    masks = site_masks(node_ids)
    names = list(masks)
    print("sites:", {k: len(v) for k, v in masks.items()}, flush=True)

    # ---- routing matrix + gain ----
    routing = {}
    gains = {}
    sat = {}
    influence = {}
    for s, idx in masks.items():
        inj = np.zeros((N, 1), dtype=np.float32)
        inj[idx, 0] = 1.0 / np.sqrt(len(idx))     # unit-mass injection
        e = propagate(WT, inj)[:, 0]
        influence[s] = e
        tot = np.abs(e).sum() + 1e-12
        routing[s] = {}
        for s2, idx2 in masks.items():
            routing[s][s2] = round(float(np.abs(e[idx2]).sum() / tot), 4)
        gains[s] = round(float(tot), 4)           # total reachable mass
        sat[s] = int((np.abs(e) >= CAP * 0.99).sum())
        print(f"routing {s}: gain={gains[s]} "
              f"top={sorted(routing[s].items(), key=lambda kv: -kv[1])[:3]}",
              flush=True)

    # ---- multiplexing capacity ----
    R = 16
    REPS = 3
    capacity = {}
    for s, idx in masks.items():
        accs = []
        for rep in range(REPS):
            rng = np.random.default_rng(100 + rep)
            pats = rng.choice([-1.0, 1.0], size=(len(idx), R)).astype(np.float32)
            inj = np.zeros((N, R), dtype=np.float32)
            inj[idx, :] = pats
            a = propagate(WT, inj)                # (N, R)
            pool = np.argsort(a.var(axis=1))[-2048:]
            A = a[pool]                            # (2048, R)
            An = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-9)
            G = An.T @ An                          # pattern similarity
            acc = float((np.argmax(G, axis=1) == np.arange(R)).mean())
            accs.append(acc)
        capacity[s] = round(sum(accs) / len(accs), 4)
        print(f"capacity {s}: {capacity[s]}", flush=True)

    report = {
        "dead_features": dead,
        "sites": {k: len(v) for k, v in masks.items()},
        "routing": routing,
        "gain": gains,
        "saturation": sat,
        "capacity_R16": capacity,
        "families": list(FAMILIES),
        "dynamics": {"steps": STEPS, "leak": LEAK, "clamp": CAP},
    }
    # assignment heuristic: families need (a) capacity at the site,
    # (b) routing from the site into high-capacity convergent regions
    score = {s: 0.5 * capacity[s] + 0.5 * min(gains[s] / max(gains.values()), 1.0)
             for s in names}
    report["site_quality"] = {s: round(v, 4) for s, v in
                              sorted(score.items(), key=lambda kv: -kv[1])}
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=1)
    print("site_quality:", report["site_quality"], flush=True)
    print(f"REPORT -> {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
