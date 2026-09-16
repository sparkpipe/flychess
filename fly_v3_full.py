#!/usr/bin/env python3
"""FliPy v3 full-brain engine with alpha-beta search and a TRAINED value head.

Value head: linear probe on the propagated activity at the readout neurons,
trained to predict Stockfish WDL values on endgame + puzzle positions.
Used as the leaf evaluation in alpha-beta (depth 4).

Move ordering: theta[slots] × activity[readout neurons] (the policy).
"""
import numpy as np, torch, scipy.sparse as sp, chess, chess.engine, time, json, sys, hashlib, os

DEV = "cuda"
_LAB = os.path.expanduser("~") + "/chess-lab"
BRAIN = _LAB + "/brain_graph.npz"
SENS = _LAB + "/sensory_idx.npy"
PUZ = "/home/spec/chess-lab/puzzles_curriculum.jsonl"
PIECES = {"P":0,"N":1,"B":2,"R":3,"Q":4,"K":5,"p":6,"n":7,"b":8,"r":9,"q":10,"k":11}
PROP_STEPS, LEAK, CAP = 2, 0.5, 20.0
N_SENSORY = 26933
N_READOUT = 4096

# ---------------- graph load ----------------
z = np.load(BRAIN)
m = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                  shape=tuple(z["shape"])).astype(np.float32)
sens = np.load(SENS)
sens_idx = sens.astype(np.int64)
N = m.shape[0]
mt = m.T.tocsr()
WT_torch = torch.sparse_csr_tensor(
    torch.from_numpy(mt.indptr.astype(np.int64)),
    torch.from_numpy(mt.indices.astype(np.int64)),
    torch.from_numpy(mt.data), size=mt.shape, device=DEV)

nprng = np.random.default_rng(20260912)
free = np.setdiff1d(np.arange(N), sens)
p2 = nprng.permutation(len(free))
readout_idx = free[p2[:N_READOUT]].astype(np.int64)
readout_t = torch.from_numpy(readout_idx).to(DEV)
sens_t = torch.from_numpy(sens_idx).to(DEV)

# ---------------- feature extraction ----------------
PIECES = {"P":0,"N":1,"B":2,"R":3,"Q":4,"K":5,"p":6,"n":7,"b":8,"r":9,"q":10,"k":11}

def extract(board, dvh=None):
    f = {}
    pts = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for t, pt in enumerate(pts):
        f[f"bb_w_{t}"] = bin(board.pieces_mask(pt, chess.WHITE)).count("1") / 8.0
        f[f"bb_b_{t}"] = bin(board.pieces_mask(pt, chess.BLACK)).count("1") / 8.0
    for sq in chess.SQUARES:
        cw = bin(board.attackers_mask(chess.WHITE, sq)).count("1")
        cb = bin(board.attackers_mask(chess.BLACK, sq)).count("1")
        f[f"atk_w_{sq}"] = min(cw, 4) / 4.0
        f[f"atk_b_{sq}"] = min(cb, 4) / 4.0
        f[f"net_{sq}"] = (min(cw,4) - min(cb,4)) / 4.0
    eyes = []
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is None: continue
        am = board.attacks_mask(sq)
        eye = np.zeros(66, dtype=np.float32)
        for s2 in chess.SQUARES:
            v = 1.0 if (am >> s2) & 1 else 0.0
            if (board.occupied_co[chess.WHITE] >> s2) & 1: v += 2.0
            if (board.occupied_co[chess.BLACK] >> s2) & 1: v += 4.0
            eye[s2] = v
        eye[64] = 1.0 if board.is_pinned(pc.color, sq) else 0.0
        eye[65] = bin(am & ~board.occupied).count("1")
        eyes.append(eye)
    for i in range(32):
        if i < len(eyes):
            for s2 in range(64): f[f"eye{i}_s{s2}"] = eyes[i][s2]
            f[f"eye{i}_pin"] = eyes[i][64]; f[f"eye{i}_mob"] = eyes[i][65]
        else:
            for s2 in range(64): f[f"eye{i}_s{s2}"] = 0.0
            f[f"eye{i}_pin"] = 0.0; f[f"eye{i}_mob"] = 0.0
    for c, ko in (("w", chess.WHITE), ("b", chess.BLACK)):
        king = board.king(ko)
        if king is None: continue
        ring = chess.BB_KING_ATTACKS[king] | (1 << king)
        for s2 in chess.SQUARES:
            if (ring >> s2) & 1:
                f[f"ks_{c}_e_{s2}"] = bin(board.attackers_mask(not ko, s2)).count("1") / 4.0
                f[f"ks_{c}_d_{s2}"] = bin(board.attackers_mask(ko, s2)).count("1") / 4.0
    f["check"] = 1.0 if board.is_check() else 0.0
    f["checkers"] = min(bin(board.checkers_mask()).count("1"), 4) / 4.0
    f["pinned"] = min(sum(1 for sq in chess.SQUARES if board.piece_at(sq)
        and board.is_pinned(board.piece_at(sq).color, sq)), 8) / 8.0
    for t, pt in enumerate(pts):
        f[f"mat_w_{t}"] = bin(board.pieces_mask(pt, chess.WHITE)).count("1") / 8.0
        f[f"mat_b_{t}"] = bin(board.pieces_mask(pt, chess.BLACK)).count("1") / 8.0
    center = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5
    f["center_w"] = min(sum(board.attackers_mask(chess.WHITE, s) for s in
        (chess.D4, chess.E4, chess.D5, chess.E5)).count("1"), 8) / 8.0
    f["center_b"] = min(sum(board.attackers_mask(chess.BLACK, s) for s in
        (chess.D4, chess.E4, chess.D5, chess.E5)).count("1"), 8) / 8.0
    mw = mb = 0
    for s in chess.SQUARES:
        pc = board.piece_at(s)
        if pc is None: continue
        n = bin(board.attacks_mask(s) & ~board.occupied).count("1")
        if pc.color == chess.WHITE: mw += n
        else: mb += n
    f["mob_w"] = min(mw, 120) / 120.0; f["mob_b"] = min(mb, 120) / 120.0
    dv = dvh or [0.0]*4
    for k in range(4): f[f"dvh_{k}"] = dv[k]
    return f

FEATURE_KEYS = None
def feat_vec(board, dvh=None):
    global FEATURE_KEYS
    f = extract(board, dvh)
    if FEATURE_KEYS is None: FEATURE_KEYS = sorted(f.keys())
    return np.array([f.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)

def move_feats(board, mv):
    g = np.zeros(8, dtype=np.float32)
    g[0] = 1.0 if board.is_capture(mv) else 0.0
    g[1] = 1.0 if board.gives_check(mv) else 0.0
    g[2] = 1.0 if mv.promotion else 0.0
    pc = board.piece_at(mv.from_square)
    g[3] = 1.0 if (pc and board.is_pinned(pc.color, mv.from_square)) else 0.0
    victim = board.piece_type_at(mv.to_square)
    g[4] = {1:1,2:3,3:3,4:5,5:9}.get(victim or 0, 0) / 9.0
    g[5] = bin(board.attackers_mask(not board.turn, mv.to_square)).count("1") / 4.0
    g[6] = bin(board.attackers_mask(board.turn, mv.to_square)).count("1") / 4.0
    g[7] = -1.0 if board.is_into_check(mv) else 0.0
    return g

def move_slot(mv):
    return mv.from_square * 64 + mv.to_square

# ---------------- model ----------------
class FlyBrain(torch.nn.Module):
    def __init__(self, n_features, n_sens, n_readout):
        super().__init__()
        self.W_sens = torch.nn.Linear(n_features, n_sens)
        self.readout_idx = torch.from_numpy(readout_idx).to(DEV)
        self.N = N
        self.theta = torch.nn.Parameter(torch.zeros(n_readout))
        self.theta_mv = torch.nn.Parameter(torch.zeros(8))
        self.value_head = torch.nn.Linear(n_readout, 1)

    def propagate(self, board, dvh=None):
        fv = feat_vec(board, dvh)
        x = torch.from_numpy(fv).to(DEV)
        s = torch.clamp(self.W_sens(x), -6, 6)
        a = torch.zeros(self.N, device=DEV)
        a[self.readout_idx * 0 + sens_t[:len(sens_t)]] = 0  # placeholder
        # inject at the actual sensory neurons
        for i, si in enumerate(sens_t[:2048]):
            a[si] = s[min(i, s.shape[0]-1)]
        for _ in range(PROP_STEPS):
            a = torch.clamp((1 - LEAK) * a + LEAK * (WT_torch @ a), -CAP, CAP)
        return a

    def scores(self, board, dvh=None):
        a = self.propagate(board, dvh)
        mvs = list(board.legal_moves)
        sc = []
        for m in mvs:
            slots, g = move_neuron_slots(m.uci())
            sc.append((self.theta[slots] * a[g]).sum())
        T = torch.stack(sc)
        MF = torch.tensor(np.stack([move_feats(board, m) for m in mvs]), device=DEV)
        T = T + MF @ self.theta_mv
        return mvs, T, a

    def value(self, board, dvh=None):
        a = self.propagate(board, dvh)
        act = a[self.readout_idx]
        return torch.sigmoid(self.value_head(act)).squeeze()

# ---------------- move neuron slots ----------------
readout_graph = readout_idx  # graph indices into the full activity vector

def move_neuron_slots(uci):
    h = hashlib.sha256(uci.encode()).digest()
    return torch.tensor([int(readout_graph[int.from_bytes(h[i:i+2], "big") % N_READOUT])
                         for i in range(0, 16, 2)], dtype=torch.int64, device=DEV)

# fix: move_neuron_slots should index into the activity at graph positions
def move_neuron_slots(uci):
    h = hashlib.sha256(uci.encode()).digest()
    return torch.tensor([int.from_bytes(h[i:i+2], "big") % N_READOUT
                         for i in range(0, 16, 2)], dtype=torch.int64, device=DEV)
