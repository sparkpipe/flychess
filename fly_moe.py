"""The mixture-of-experts fly (operator design 2026-09-19): shared
connectome mixing (ruled fine), graded gating (the dopamine analog),
per-expert injection doors + readout thetas. The gate is a small sigmoid
head per regime axis on the feature vector — co-active by design
(attack AND defense). Expert k's gradient is scaled by its gate mass:
automatic specialization without hard routing. Borderline positions
train multiple experts simultaneously.

Self-contained trainer (DEGM-only first job) with the house cadence:
500-step blocks, EXHAUSTIVE gates, FAM_SCORE-weighted sampling.
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
from fly_curriculum import (DEV, LEAK, PROP_STEPS, load_pools,
                            FAM_SCORE)

STATE = os.environ.get("MSTATE",
                       "/home/spec/chess-lab/fly_moe_degm.pt")
N_EXP = int(os.environ.get("N_EXP", "6"))
INJ_K = 4096
CAP = 6.0
B = 32


class MoEFly(nn.Module):
    def __init__(self, n_feats, template):
        super().__init__()
        self.N = template.N
        self.WT = template.WT              # shared frozen mixing
        self.readout_idx = template.readout_idx
        self.inj_sets = []
        g = np.random.default_rng(99)
        pool = g.permutation(self.N)[: N_EXP * INJ_K]
        for k in range(N_EXP):
            self.inj_sets.append(torch.from_numpy(
                pool[k * INJ_K:(k + 1) * INJ_K].astype(np.int64)).to(DEV))
        self.gate = nn.Sequential(
            nn.Linear(n_feats, 64), nn.ReLU(),
            nn.Linear(64, N_EXP))
        self.enc = nn.ModuleList(
            [nn.Linear(n_feats, INJ_K) for _ in range(N_EXP)])
        for e in self.enc:
            nn.init.normal_(e.weight, std=0.05)
            nn.init.zeros_(e.bias)
        self.theta = nn.Parameter(
            torch.ones(N_EXP, 4096, device=DEV) * 0.1)
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
        self.cls_idx = template.cls_idx
        self.retino = None
        self.wmask = None

    def gate_w(self, x):
        return torch.sigmoid(self.gate(x))   # (B, N_EXP), co-active

    def logits_all(self, a):
        th = self.theta.sum(0)
        return th.unsqueeze(0) * a[self.readout_idx].T

    def propagate(self, fvb, reach=None):
        x = torch.from_numpy(fvb).to(DEV)
        g = self.gate_w(x)                    # (B, K)
        B = x.shape[0]
        a = torch.zeros(self.N, B, device=DEV)
        for k in range(N_EXP):
            s = torch.clamp(self.enc[k](x), -CAP, CAP)
            a[self.inj_sets[k]] = a[self.inj_sets[k]] + \
                (s * g[:, k:k + 1]).T
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
        return a, g


def forward_moe(model, fvb, slotb, pcrowb, mfb, maskb, pseudob=None,
                threatb=None, pseudo2b=None):
    a, g = model.propagate(fvb)
    Bn = a.shape[1]
    cols = torch.arange(Bn, device=DEV).unsqueeze(1)
    slots = torch.from_numpy(slotb).to(DEV)
    geo = model.geo_w(model.slot_geo).squeeze(-1)
    mf_t = torch.from_numpy(mfb).to(DEV)
    act = a[model.readout_idx.cpu().numpy()][:, :, None] \
        if False else None
    ri = model.readout_idx
    A = a[ri]                                  # (4096, B)
    base = geo[slots] + mf_t @ model.theta_mv + \
        (mf_t * model.theta_mv_pc[
            torch.from_numpy(pcrowb).to(DEV)]).sum(-1)
    if pseudob is not None:
        base = base + model.w_pseudo * torch.from_numpy(pseudob).to(DEV)
    if threatb is not None:
        base = base + model.w_threat * torch.from_numpy(threatb).to(DEV)
    if pseudo2b is not None:
        base = base + model.w_pseudo2 * torch.from_numpy(pseudo2b).to(DEV)
    # per-expert readout, mixed by the gate: theta_k[slot]*act[slot,b]
    T = torch.zeros(slots.shape, device=DEV)
    actslots = A[slots]                        # (B, M, 4096)? no: A is
    for k in range(N_EXP):
        tk = model.theta[k]
        T = T + g[:, k:k + 1] * (tk[slots] * A.T[cols, slots])
    T = T + base
    T = T.masked_fill(~torch.from_numpy(maskb).to(DEV), -1e9)
    return T, torch.log_softmax(T, dim=1), None, None


def build_batch_moe(rng, rows, batch):
    buf = []
    pools = [e.get("pool", "?") for e in rows]
    wts = [max(0.05, 1.0 - FAM_SCORE.get(pl, 0.0)) for pl in pools]
    cum, t = [], 0.0
    for w in wts:
        t += w
        cum.append(t)
    import bisect

    def pick():
        r = rng.random() * t
        return rows[bisect.bisect_left(cum, r)]
    while len(buf) < batch:
        e = pick()
        try:
            b = chess.Board(e["fen"])
            if b.is_game_over():
                continue
            buf.append((b, e))
        except Exception:
            continue
    fvb = np.stack([flyfeat_cb.feat_vec(b)[0] for b, _ in buf])
    clb = np.array([fc.CLS_MAP.get(e["cat"], 1) for _, e in buf],
                   dtype=np.int64)
    rich, slots, tgts = [], [], []
    for b, e in buf:
        mvs = list(b.legal_moves)
        ch = e.get("children", {})
        tv = fc.graded_targets(e, b)
        sv, vv, vm, best = [], [], [], None
        p2 = {(pm.from_square, pm.to_square)
              for pm in b.pseudo_legal_moves}
        mfs, pss, ps2s, thrs, pci = [], [], [], [], []
        for k, mv in enumerate(mvs):
            u = mv.uci()
            sv.append(mv.from_square * 64 + mv.to_square)
            vv.append(tv.get(u, 0.0))
            vm.append(u in tv)
            if u == e["best"]:
                best = k
            mfs.append(flyfeat_cb.move_feats(b, mv))
            pss.append(1.0 if (b.attacks_mask(mv.from_square)
                               & chess.BB_SQUARES[mv.to_square])
                       else 0.0)
            ps2s.append(1.0 if (mv.from_square, mv.to_square) in p2
                        else 0.0)
            pc = b.piece_at(mv.from_square)
            pci.append(fc._PC_IDX[pc.piece_type] if pc else 0)
            b.push(mv)
            thrs.append(min(bin(b.attacks_mask(mv.to_square)
                                & b.occupied_co[b.turn]).count("1"),
                            4) / 4.0)
            b.pop()
        rich.append((np.stack(mfs), np.array(pss, np.float32),
                     np.array(ps2s, np.float32),
                     np.array(thrs, np.float32),
                     np.array(pci, np.int64)))
        slots.append((np.array(sv, dtype=np.int64),
                      np.array(vv, dtype=np.float32),
                      np.array(vm, dtype=bool)))
        tgts.append(best if best is not None else -1)
    Bn = fvb.shape[0]
    M = max(len(s[0]) for s in slots)
    F = rich[0][0].shape[1]
    slotb = np.zeros((Bn, M), np.int64)
    pcrowb = np.zeros((Bn, M), np.int64)
    mfb = np.zeros((Bn, M, F), np.float32)
    maskb = np.zeros((Bn, M), bool)
    vmask = np.zeros((Bn, M), bool)
    pseudob = np.zeros((Bn, M), np.float32)
    threatb = np.zeros((Bn, M), np.float32)
    pseudo2b = np.zeros((Bn, M), np.float32)
    tv = np.zeros((Bn, M), np.float32)
    for i, (sv, vv, vm), (mfs, pss, ps2s, thrs, pci) in zip(
            range(Bn), slots, rich):
        L = len(sv)
        slotb[i, :L] = sv
        pcrowb[i, :L] = pci
        mfb[i, :L] = mfs
        maskb[i, :L] = True
        vmask[i, :L] = vm
        pseudob[i, :L] = pss
        threatb[i, :L] = thrs
        pseudo2b[i, :L] = ps2s
        tv[i, :L] = vv
    return (fvb, slotb, pcrowb, mfb, maskb, pseudob, threatb, pseudo2b,
            tv, vmask, np.array(tgts, np.int64), clb)


def tb_step_moe(model, opt, rows, rng):
    (fvb, slotb, pcrowb, mfb, maskb, pseudob, threatb, pseudo2b,
     tv, vmask, tgt, cl) = build_batch_moe(rng, rows, B)
    T, logp, _, _ = forward_moe(model, fvb, slotb, pcrowb, mfb, maskb,
                                pseudob, threatb, pseudo2b)
    msk = torch.from_numpy(maskb).to(DEV)
    tvv = torch.from_numpy(tv).to(DEV)
    vmk = torch.from_numpy(vmask).to(DEV)
    hub = torch.nn.functional.huber_loss(T, tvv, delta=0.5,
                                         reduction="none")
    loss_val = hub[vmk].mean()
    t = torch.from_numpy(tgt).to(DEV)
    has = t >= 0
    if has.any():
        loss_m = torch.nn.functional.cross_entropy(
            logp[has], t[has], ignore_index=-1)
    else:
        loss_m = torch.zeros((), device=DEV)
    loss = loss_val * 2.0 + loss_m
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())


def gate_moe(model, rows):
    model.eval()
    ok = tot = 0
    fam = {}
    by_pool = {}
    for e in rows:
        by_pool.setdefault(e.get("pool", "?"), []).append(e)
    FLIP = {"win": "loss", "loss": "win", "draw": "draw",
            "cursed_win": "cursed_loss", "cursed_loss": "cursed_win"}
    with torch.no_grad():
        for pn, prows in by_pool.items():
            fst = fam.setdefault(pn, [0, 0])
            for ci in range(0, len(prows), 256):
                chunk = prows[ci:ci + 256]
                keep = []
                for e in chunk:
                    try:
                        b = chess.Board(e["fen"])
                    except Exception:
                        continue
                    if b.is_game_over() or not list(b.legal_moves):
                        continue
                    keep.append(e)
                if not keep:
                    continue
                Bn = len(keep)
                boards = [chess.Board(e["fen"]) for e in keep]
                M = max(len(list(b.legal_moves)) for b in boards)
                F = flyfeat_cb.MOVE_DIMS
                fvb = np.stack([flyfeat_cb.feat_vec(b)[0]
                                for b in boards])
                slotb = np.zeros((Bn, M), np.int64)
                pcrowb = np.zeros((Bn, M), np.int64)
                mfb = np.zeros((Bn, M, F), np.float32)
                maskb = np.zeros((Bn, M), bool)
                psb = np.zeros((Bn, M), np.float32)
                ps2b = np.zeros((Bn, M), np.float32)
                thb = np.zeros((Bn, M), np.float32)
                for i, b in enumerate(boards):
                    mvs = list(b.legal_moves)
                    p2 = {(pm.from_square, pm.to_square)
                          for pm in b.pseudo_legal_moves}
                    for j, mv in enumerate(mvs):
                        slotb[i, j] = mv.from_square * 64 + mv.to_square
                        pc = b.piece_at(mv.from_square)
                        pcrowb[i, j] = fc._PC_IDX[pc.piece_type] \
                            if pc else 0
                        mfb[i, j] = flyfeat_cb.move_feats(b, mv)
                        maskb[i, j] = True
                        psb[i, j] = 1.0 if (
                            b.attacks_mask(mv.from_square)
                            & chess.BB_SQUARES[mv.to_square]) else 0.0
                        ps2b[i, j] = 1.0 if (mv.from_square,
                                             mv.to_square) in p2 else 0.0
                        b.push(mv)
                        thb[i, j] = min(
                            bin(b.attacks_mask(mv.to_square)
                                & b.occupied_co[b.turn]).count("1"),
                            4) / 4.0
                        b.pop()
                T, logp, _, _ = forward_moe(model, fvb, slotb, pcrowb,
                                            mfb, maskb, psb, thb, ps2b)
                picks = torch.argmax(T, dim=1).tolist()
                for i, e in enumerate(keep):
                    ch = e.get("children", {})
                    opt = {"win": "loss", "cursed_win": "loss",
                           "draw": "draw", "cursed_loss": "win",
                           "loss": "win"}[e["cat"]]
                    optset = {u for u, c in ch.items()
                              if FLIP.get(c.get("cat")) == opt}
                    mvs = list(boards[i].legal_moves)
                    pick = mvs[picks[i]].uci()
                    tot += 1
                    fst[1] += 1
                    hit = pick in optset
                    fst[0] += int(hit)
                    ok += int(hit)
    model.train()
    famtbl = {nm: round(fok / max(ftot, 1), 3)
              for nm, (fok, ftot) in sorted(fam.items())}
    return ok / max(tot, 1), famtbl


def main():
    torch.manual_seed(0)
    flyfeat_cb.feat_vec(chess.Board())
    rmap = fc.build_retino_map(mode="geo")
    template = fc.FlyCB(len(flyfeat_cb.FEATURE_KEYS), sel_boards=None,
                        readout="variance").to(DEV)
    template.retino = rmap
    template.retino_gain = torch.nn.Parameter(
        torch.ones(7) * 2.0).to(DEV)
    m = MoEFly(len(flyfeat_cb.FEATURE_KEYS), template).to(DEV)
    if os.path.exists(STATE) and os.environ.get("FRESH", "0") != "1":
        m.load_state_dict(torch.load(STATE, weights_only=True),
                          strict=False)
        print("resumed", flush=True)
    else:
        print(f"FRESH MoE fly ({N_EXP} experts)", flush=True)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    rows = load_pools([f"DEGM_Ch{i}" for i in range(1, 16)])
    rng = random.Random(6000)
    step = 0
    gw = []
    while True:
        for _ in range(500):
            step += 1
            tb_step_moe(m, opt, rows, rng)
        torch.save(m.state_dict(), STATE + ".tmp")
        os.replace(STATE + ".tmp", STATE)
        pair, fam = gate_moe(m, rows)
        worst = min(fam.values()) if fam else 0.0
        FAM_SCORE.update(fam)
        print(json.dumps({"moe_fly": True, "step": step,
                          "opt_set": round(pair, 4),
                          "worst": worst}), flush=True)
        print("S6 FAM " + " ".join(f"{k}={v}" for k, v in fam.items()),
              flush=True)
        if pair >= 0.98 and worst >= 0.98:
            print("MOE-FLY DEGM PASSED (exhaustive)", flush=True)
            break


if __name__ == "__main__":
    main()
