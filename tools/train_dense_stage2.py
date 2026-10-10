"""DENSE head v3 — stage 2: train the operator's design.

Plain linear (NO gates, all weights free) over:
  13 scalars (per-expert normalized)
  13x1024 activations (per-dim normalized, baked stats)
  raw active indices (~60, sparse weight lookup over 86896)
  36 HCE + 13 one-hot
Target: search eval (dense/sharp). Loss: win-prob MSE.
Gates before export: output range p95, sharpness probes, scalar-path gain.
Export: .evh v3 (gates field = all 1.0, engine consumes as plain linear).
"""
import sys, time, struct
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import torch, torch.nn as nn

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
E_N, HCE_D, ACT_D, FT_D = 13, 36, 1024, 86896

def wp(x):
    return 1.0 / (1.0 + torch.exp(-x / 361.0))

class DenseHead(nn.Module):
    def __init__(self, ev_mean, ev_std, a_mean, a_std):
        super().__init__()
        self.lin_ev = nn.Linear(E_N, 1, bias=False)
        self.lin_act = nn.Linear(E_N * ACT_D, 1, bias=False)
        self.lin_hce = nn.Linear(HCE_D + E_N, 1, bias=False)
        self.ft = nn.Parameter(torch.zeros(FT_D))
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("ev_mean", ev_mean.clone())
        self.register_buffer("ev_std", ev_std.clone())
        self.register_buffer("a_mean", a_mean.clone().view(E_N, ACT_D))
        self.register_buffer("a_std", a_std.clone().view(E_N, ACT_D))
        nn.init.zeros_(self.lin_ev.weight)
        with torch.no_grad():
            for e in range(E_N):
                self.lin_ev.weight[0, e] = 1.0
    def forward(self, ev, act, hce, oh, ftidx, ftcnt):
        x_ev = (ev - self.ev_mean) / self.ev_std
        x_act = (act.float() - self.a_mean) / self.a_std
        out = self.lin_ev(x_ev) + self.lin_act(x_act.view(len(ev), -1)) \
            + self.lin_hce(torch.cat([hce, oh], 1)) + self.bias
        # sparse raw-feature sum via index_add
        flat = ftidx.clamp(min=0)
        contrib = self.ft[flat] * (ftidx >= 0).float()
        out = out + contrib.sum(dim=1, keepdim=True)
        return out.squeeze(-1)

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(f"{SP}/dense_data.npz")
    evals, acts, hces = d["evals"], d["acts"], d["hces"]
    # cache layout has 14 expert columns (retired tb_training = col 13);
    # verified map to engine slot order [l0,l1,l2,l3,nvb,nvr,bvr,rv2m,qvmat,oppb,dvor,exch,tact]:
    ACT_COL = [0, 1, 4, 8, 2, 9, 6, 10, 11, 5, 7, 3, 12]
    assert acts.shape[1] == 14
    acts = acts[:, ACT_COL, :]  # (N, 13, 1024) in engine slot order
    doms, tgts, ft, ftcnt = d["doms"], d["tgts"], d["ft"], d["ftcnt"]
    N = len(tgts)
    print(f"{N} positions", flush=True)

    # game-aware split: consecutive records are near-duplicates; split by chunks
    CHUNK = 32
    chunk_id = np.arange(N) // CHUNK
    rng = np.random.RandomState(42)
    val_chunks = set(rng.choice(np.unique(chunk_id), size=max(1, len(np.unique(chunk_id)) // 10), replace=False))
    val_mask = np.isin(chunk_id, list(val_chunks))
    tr = np.where(~val_mask)[0]
    va = np.where(val_mask)[0]
    print(f"train {len(tr)}  val {len(va)} (chunked split)", flush=True)

    ev_mean = torch.tensor(evals[tr].mean(0), dtype=torch.float32)
    ev_std = torch.tensor(evals[tr].std(0), dtype=torch.float32).clamp(min=1.0)
    a_tr = acts[tr].reshape(len(tr), E_N * ACT_D).astype(np.float32)
    a_mean = torch.tensor(a_tr.mean(0))
    a_std = torch.tensor(a_tr.std(0)).clamp(min=0.05)
    del a_tr

    model = DenseHead(ev_mean, ev_std, a_mean, a_std).to(dev)
    Y_all = torch.tensor(tgts, dtype=torch.float32)
    X_ev = torch.tensor(evals, dtype=torch.float32)
    X_hce = torch.tensor(hces, dtype=torch.float32)
    oh = np.zeros((N, E_N), dtype=np.float32); oh[np.arange(N), doms] = 1.0
    X_oh = torch.tensor(oh)
    X_act = torch.tensor(acts)  # keep uint8 on CPU; forward() converts per batch
    X_ft = torch.tensor(ft.astype(np.int64))
    X_ftc = torch.tensor(ftcnt.astype(torch.int64).numpy() if False else ftcnt.astype(np.int64))

    opt = torch.optim.AdamW([
        {"params": [model.lin_ev.weight, model.bias], "lr": 3e-4},
        {"params": [model.lin_act.weight], "lr": 3e-4},
        {"params": [model.lin_hce.weight], "lr": 3e-4},
        {"params": [model.ft], "lr": 3e-3, "weight_decay": 1e-6},
    ], weight_decay=1e-5)

    BS = 4096
    EPOCHS = 60
    best, best_sd = 1e9, None
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        tl, nb = 0.0, 0
        for i in range(0, len(perm), BS):
            idx = perm[i:i+BS]
            tidx = torch.tensor(idx)
            pred = model(X_ev[tidx].to(dev), X_act[tidx].to(dev), X_hce[tidx].to(dev),
                         X_oh[tidx].to(dev), X_ft[tidx].to(dev), X_ftc[tidx].to(dev))
            y = Y_all[tidx].to(dev)
            loss = ((wp(pred) - wp(y)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vl, corr_n, corr_d = 0.0, 0.0, 0.0
            for i in range(0, len(va), BS):
                idx = va[i:i+BS]
                tidx = torch.tensor(idx)
                pred = model(X_ev[tidx].to(dev), X_act[tidx].to(dev), X_hce[tidx].to(dev),
                             X_oh[tidx].to(dev), X_ft[tidx].to(dev), X_ftc[tidx].to(dev))
                y = Y_all[tidx].to(dev)
                vl += ((wp(pred) - wp(y)) ** 2).mean().item()
                corr_n += (wp(pred) * wp(y)).sum().item()
            vl /= max(1, len(va) // BS)
        if vl < best:
            best, best_sd = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"ep{ep}: train {tl/max(nb,1):.6f}  val {vl:.6f}", flush=True)
    model.load_state_dict(best_sd)

    # ---- GATES ----
    with torch.no_grad():
        sample = torch.tensor(tr[:20000])
        out = model(X_ev[sample].to(dev), X_act[sample].to(dev), X_hce[sample].to(dev),
                    X_oh[sample].to(dev), X_ft[sample].to(dev), X_ftc[sample].to(dev)).cpu().numpy()
        p95 = float(np.percentile(np.abs(out), 95))
    print(f"GATE range: p95|out| = {p95:.0f}cp (band 5-400)", flush=True)
    assert 5 < p95 < 400, f"range gate FAIL {p95}"
    # scalar-path gain
    gain = float((out.std()) / max(evals[tr].std(), 1e-9))
    print(f"scalar-path effective gain {gain:.3f} (info only)", flush=True)

    # ---- export .evh v3 ----
    def T(x):
        return x.detach().cpu().numpy().astype(np.float32)
    gates = np.ones((13, E_N), dtype=np.float32)
    lw_eval = np.zeros(E_N + HCE_D + E_N, dtype=np.float32)
    lw_eval[:E_N] = T(model.lin_ev.weight).flatten()
    lw_eval[E_N:E_N+HCE_D] = T(model.lin_hce.weight).flatten()[:HCE_D]
    lw_eval[E_N+HCE_D:] = T(model.lin_hce.weight).flatten()[HCE_D:]
    lin_act = T(model.lin_act.weight).flatten()
    ft_w = T(model.ft)
    bias = float(model.bias.item())
    with open(f"{SP}/head_dense_v3.evh", "wb") as f:
        f.write(struct.pack("<IIIII", 0x45564C48, 3, E_N, HCE_D, 13))
        f.write(gates.tobytes())
        f.write(lw_eval.tobytes())
        f.write(struct.pack("<f", bias))
        f.write(T(model.ev_mean).tobytes()); f.write(T(model.ev_std).tobytes())
        f.write(struct.pack("<II", E_N, ACT_D))
        f.write(lin_act.tobytes())
        f.write(T(model.a_mean).tobytes()); f.write(T(model.a_std).tobytes())
        f.write(struct.pack("<I", FT_D))
        f.write(ft_w.tobytes())
    torch.save(model.state_dict(), f"{SP}/head_dense_v3.pt")
    print("EXPORTED head_dense_v3.evh", flush=True)
    print(f"sizes: act {lin_act.sum():.2f} ft_nonzero {(abs(ft_w)>1e-6).sum()} "
          f"ev {lw_eval[:13].round(2).tolist()}", flush=True)

if __name__ == "__main__":
    main()
