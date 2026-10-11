"""The expanded fly (operator design 2026-09-19): duplicate the saturated
core. The saturation census showed movement/concepts/DEGM share the SAME
top-20K load-bearing neurons (97% overlap) — all training competes for
one core. This doubles that area: WT2 = N originals + K clones of the
top-K load-bearing neurons; clones mirror incoming edges (in-S sources
remapped to clones, outside sources shared). Original path frozen; the
clone path (fresh sensory encoder + fresh readout bank) trains DEGM.
"""
import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import chess

sys.path.insert(0, "/home/spec/chess-lab")
import fly_curriculum as fc
import flyfeat_cb
from fly_curriculum import (DEV, LEAK, PROP_STEPS, forward, gate_tb,
                            load_pools, FAM_SCORE)

STATE = os.environ.get("XSTATE",
                       "/home/spec/chess-lab/fly_expanded_degm.pt")
SRC = "/home/spec/chess-lab/fly_cb_v3_s6.pt"
K = int(os.environ.get("EXPAND_K", "20000"))
N_INJ = 8192
CAP = 6.0


def build_wt2(src_model, topk_idx):
    """Clones mirror incoming edges; sources inside S remap to clones."""
    wt = src_model.WT.cpu()
    crow = wt.crow_indices().numpy()
    col = wt.col_indices().numpy().astype(np.int64)
    val = wt.values().numpy()
    N = wt.shape[0]
    K = len(topk_idx)
    clone_of = -np.ones(N, dtype=np.int64)
    clone_of[topk_idx] = np.arange(N, N + K)
    inS = np.zeros(N + K, dtype=bool)
    inS[topk_idx] = True
    inS[N:] = True
    rows2, cols2, vals2 = [], [], []
    nnz2 = 0
    for v in range(N):
        s, e = int(crow[v]), int(crow[v + 1])
        rows2.append((col[s:e], val[s:e]))
        nnz2 += e - s
    for v in topk_idx:
        s, e = int(crow[v]), int(crow[v + 1])
        src = col[s:e].copy()
        remapped = np.where(clone_of[src] >= 0, clone_of[src], src)
        rows2.append((remapped, val[s:e]))
        nnz2 += e - s
    cols = np.concatenate([r[0] for r in rows2])
    vals = np.concatenate([r[1] for r in rows2])
    crows = np.zeros(N + K + 1, dtype=np.int64)
    crows[1:] = np.cumsum([len(r[0]) for r in rows2])
    WT2 = torch.sparse_csr_tensor(
        torch.from_numpy(crows.astype(np.int32)).to(DEV),
        torch.from_numpy(cols.astype(np.int32)).to(DEV),
        torch.from_numpy(vals.astype(np.float32)).to(DEV),
        size=(N + K, N + K))
    return WT2, N, K


class ExpandedFly(nn.Module):
    def __init__(self, n_feats, src_model, topk_idx):
        super().__init__()
        self.WT, self.N_orig, self.K = build_wt2(src_model, topk_idx)
        self.N = self.N_orig + self.K
        self.src = src_model
        # clone-community injection + readout bank
        g = np.random.default_rng(7)
        self.inj_idx = torch.from_numpy(
            np.arange(self.N_orig,
                      self.N_orig + N_INJ, dtype=np.int64)).to(DEV)
        ro_src = src_model.readout_idx.cpu().numpy()
        clone_of = -np.ones(self.N_orig, dtype=np.int64)
        clone_of[topk_idx] = np.arange(self.N_orig, self.N)
        ro2 = clone_of[ro_src]
        ro2 = np.where(ro2 >= 0, ro2,
                       self.N_orig + (ro_src % self.K))
        self.readout_idx = torch.from_numpy(ro2).to(DEV)
        self.W_sens = nn.Linear(n_feats, N_INJ)
        nn.init.normal_(self.W_sens.weight, std=0.05)
        nn.init.zeros_(self.W_sens.bias)
        self.theta = nn.Parameter(torch.ones(4096, device=DEV) * 0.1)
        self.slot_geo = nn.Parameter(torch.zeros(4096, 7, device=DEV))
        self.geo_w = nn.Linear(7, 1)
        self.theta_mv = nn.Parameter(torch.zeros(
            flyfeat_cb.MOVE_DIMS, device=DEV))
        self.theta_mv_pc = nn.Parameter(torch.zeros(
            6, flyfeat_cb.MOVE_DIMS, device=DEV))
        self.w_pseudo = nn.Parameter(torch.tensor(0.0, device=DEV))
        self.w_pseudo2 = nn.Parameter(torch.tensor(0.0, device=DEV))
        self.w_threat = nn.Parameter(torch.tensor(0.0, device=DEV))
        self.theta_cls = nn.Parameter(torch.zeros(3, device=DEV))
        self.cls_idx = torch.from_numpy(
            (ro2[:3]).copy()).to(DEV)
        self.retino = None
        self.wmask = None

    def logits_all(self, a):
        return self.theta.unsqueeze(0) * a[self.readout_idx].T

    def propagate(self, fvb, reach=None):
        x = torch.from_numpy(fvb).to(DEV)
        B = x.shape[0]
        s = torch.clamp(self.W_sens(x), -CAP, CAP)
        a = torch.zeros(self.N, B, device=DEV)
        a[self.inj_idx] = s.T
        eps = float(os.environ.get("MEPS", "0.02"))
        _prev = None
        for _ in range(int(os.environ.get("MPROP",
                                          str(PROP_STEPS)))):
            a = (1 - LEAK) * a + LEAK * (self.WT @ a)
            a = a / (a.abs().mean() + 1e-6) * 2.0
            if _ % 4 == 3 and _prev is not None and float(
                    (a - _prev).abs().mean()) < eps:
                break
            if _ % 4 == 3:
                _prev = a
        return a


def main():
    torch.manual_seed(0)
    flyfeat_cb.feat_vec(chess.Board())
    rmap = fc.build_retino_map(mode="geo")
    src = fc.FlyCB(len(flyfeat_cb.FEATURE_KEYS), sel_boards=None,
                   readout="variance").to(DEV)
    src.retino = rmap
    src.retino_gain = torch.nn.Parameter(
        torch.ones(7) * 2.0).to(DEV)
    src.load_state_dict(torch.load(SRC, weights_only=True),
                        strict=False)
    src.eval()
    topk = np.load("/tmp/fly_saturated_top20k.npy")
    m = ExpandedFly(len(flyfeat_cb.FEATURE_KEYS), src, topk).to(DEV)
    if os.path.exists(STATE) and os.environ.get("FRESH", "0") != "1":
        m.load_state_dict(torch.load(STATE, weights_only=True),
                          strict=False)
        print("resumed", flush=True)
    else:
        print("FRESH expanded fly (K clones of the saturated core)",
              flush=True)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    rows = load_pools([f"DEGM_Ch{i}" for i in range(1, 16)])
    rng = random.Random(6000)
    step = 0
    while True:
        for _ in range(500):
            step += 1
            fc.tb_step(m, opt, rows, rng)
        torch.save(m.state_dict(), STATE + ".tmp")
        os.replace(STATE + ".tmp", STATE)
        pair, _, fam = gate_tb(m, rows, random.Random(777),
                               exhaustive=True)
        worst = min(fam.values()) if fam else 0.0
        FAM_SCORE.update(fam)
        print(json.dumps({"expanded_fly": True, "step": step,
                          "opt_set": round(pair, 4),
                          "worst": worst}), flush=True)
        print("S6 FAM " + " ".join(f"{k}={v}" for k, v in fam.items()),
              flush=True)
        if pair >= 0.98 and worst >= 0.98:
            print("EXPANDED-FLY DEGM PASSED (exhaustive)", flush=True)
            break


if __name__ == "__main__":
    main()
