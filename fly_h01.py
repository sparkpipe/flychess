"""The H01 graft: the human-connectome substrate trained as a fly-class
model. Loads the snowball subgraph (h01_graph.npz), wires a trainable
sensory encoder into hub neurons, propagates with the fly's own dynamics
(leaky settling + ANORM renorm), and reads out through 4096 slot neurons
— the SAME attribute interface as FlyCB, so fly_curriculum's forward(),
tb_step, and gate_tb drive it unchanged. DEGM-only training (the graft's
first job: hold the book where the fly plateaus).
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

GRAPH = "/home/spec/chess-lab/h01_graph.npz"
STATE = os.environ.get("HSTATE", "/home/spec/chess-lab/fly_h01_degm.pt")
N_INJ = 8192
CAP = 6.0


class H01Fly(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        d = np.load(GRAPH)
        n = int(d["n"])
        u, v, sgn = d["u"], d["v"], d["sign"]
        self.N = n
        self._u_host = u
        self._v_host = v
        self._init_sparse()
        deg = np.bincount(v, minlength=n) + np.bincount(u, minlength=n)
        hubs = np.argsort(-deg)
        g = torch.Generator().manual_seed(123)
        inj = hubs[:N_INJ]
        self.inj_idx = torch.from_numpy(inj.copy()).to(DEV)
        ro = np.zeros(4096, dtype=np.int64)
        pool = hubs[N_INJ:N_INJ + 32768]
        ro[:] = pool[np.random.default_rng(5).choice(len(pool), 4096,
                                                    replace=True)]
        self.readout_idx = torch.from_numpy(ro).to(DEV)
        self.W_sens = nn.Linear(n_feats, len(inj))
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
        self.theta_cls = nn.Parameter(torch.zeros(3, 4096, device=DEV))
        self.cls_idx = self.readout_idx
        self.retino = None
        self.wmask = None

    def _init_sparse(self):
        # build CSR properly: rows = v (target), cols = u, vals = sign
        order = np.argsort(self._v_host, kind="stable")
        v_s = self._v_host[order]
        u_s = self._u_host[order]
        s_s = None
        indptr = np.zeros(self.N + 1, dtype=np.int64)
        np.add.at(indptr[1:], np.bincount(v_s, minlength=self.N), 0)
        cnt = np.bincount(v_s, minlength=self.N)
        indptr = np.concatenate([[0], np.cumsum(cnt).astype(np.int64)])
        d = np.load(GRAPH)
        s_s = d["sign"][order]
        self.WT = torch.sparse_csr_tensor(
            torch.from_numpy(indptr.astype(np.int32)).to(DEV),
            torch.from_numpy(u_s.astype(np.int32)).to(DEV),
            torch.from_numpy(s_s.astype(np.float32)).to(DEV),
            size=(self.N, self.N))

    def logits_all(self, a):
        return self.theta.unsqueeze(0) * a[self.readout_idx].T

    def propagate(self, fvb, reach=None):
        x = torch.from_numpy(fvb).to(DEV)
        B = x.shape[0]
        s = torch.clamp(self.W_sens(x), -6, 6)
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
    m = H01Fly(len(flyfeat_cb.FEATURE_KEYS)).to(DEV)
    if os.path.exists(STATE) and os.environ.get("FRESH", "0") != "1":
        m.load_state_dict(torch.load(STATE, weights_only=True),
                          strict=False)
        print("resumed", flush=True)
    else:
        print("FRESH graft", flush=True)
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
        pair, tot_, fam = gate_tb(m, rows, random.Random(777),
                                  exhaustive=True)
        worst = min(fam.values()) if fam else 0.0
        FAM_SCORE.update(fam)
        print(json.dumps({"h01_graft": True, "step": step,
                          "opt_set": round(pair, 4),
                          "worst": worst}), flush=True)
        print("S6 FAM " + " ".join(f"{k}={v}" for k, v in fam.items()),
              flush=True)
        if pair >= 0.98 and worst >= 0.98:
            print("H01-GRAFT DEGM PASSED (exhaustive)", flush=True)
            break


if __name__ == "__main__":
    main()
