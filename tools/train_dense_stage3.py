"""DENSE NONLINEAR head v4 — 14th accumulator + NNUE-style joint tail.

Inputs (14,398 dense):
  13 x 1024 frozen-expert states (engine scale: cache/2)
  1024 raw-board learned embedding (86,896-vocab accumulator, trained fresh)
  13 scalars (each expert's own 32->32->1 output, normalized)
  36 HCE + 13 one-hot
Head: Linear(14398, 64) -> clipped ReLU -> Linear(64, 32) -> clipped ReLU -> Linear(32,1)
Target: search eval (wp loss). Train until val stops improving (patience),
checkpoint EVERY epoch; best-val checkpoint kept separately.
Export: .evh v4 quantized (emb int16, W1/W2 int16, out int32) — after gates.
"""
import sys, os, time, glob
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import torch, torch.nn as nn

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
E_N, HCE_D, ACT_D, FT_D = 13, 36, 1024, 86896
H1, H2 = 64, 32
CKPT_DIR = f"{SP}/dense_v4_ckpts3"

def wp(x):
    return 1.0 / (1.0 + torch.exp(-x / 361.0))

class ClippedReLU(nn.Module):
    def __init__(self, hi=63.0):
        super().__init__()
        self.hi = hi
    def forward(self, x):
        return torch.clamp(x, 0.0, self.hi)

class DenseV4(nn.Module):
    def __init__(self, ev_mean, ev_std, a_mean, a_std):
        super().__init__()
        self.emb = nn.Embedding(FT_D, ACT_D, sparse=True)
        nn.init.normal_(self.emb.weight, std=0.01)
        self.lin_ev = nn.Linear(E_N, 1, bias=False)
        nn.init.zeros_(self.lin_ev.weight)
        with torch.no_grad():
            for e in range(E_N):
                self.lin_ev.weight[0, e] = 1.0
        self.tail = nn.Sequential(
            nn.Linear(E_N * ACT_D + ACT_D + HCE_D + E_N, H1), ClippedReLU(),
            nn.Linear(H1, H2), ClippedReLU(),
            nn.Linear(H2, 1),
        )
        self.register_buffer("ev_mean", ev_mean.clone())
        self.register_buffer("ev_std", ev_std.clone())
        self.register_buffer("a_mean", a_mean.clone().view(E_N, ACT_D))
        self.register_buffer("a_std", a_std.clone().view(E_N, ACT_D))

    def forward(self, ev, act, hce, oh, ftidx):
        x_ev = (ev - self.ev_mean) / self.ev_std
        x_act = (act.float() - self.a_mean) / self.a_std
        mask = (ftidx >= 0).float().unsqueeze(-1)
        raw = (self.emb(ftidx.clamp(min=0)) * mask).sum(dim=1)  # (B, 1024), pads zeroed
        feats = torch.cat([x_act.view(len(ev), -1), raw, hce, oh], dim=1)
        return self.tail(feats).squeeze(-1) + self.lin_ev(x_ev).squeeze(-1)

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    os.makedirs(CKPT_DIR, exist_ok=True)
    d = np.load(f"{SP}/dense_data.npz")
    evals, acts, hces = d["evals"], d["acts"], d["hces"]
    doms, tgts, ft = d["doms"], d["tgts"], d["ft"]
    ACT_COL = [0, 1, 4, 8, 2, 9, 6, 10, 11, 5, 7, 3, 12]
    assert acts.shape[1] == 14
    acts = np.ascontiguousarray(acts[:, ACT_COL, :]) >> 1     # engine scale (cache/2)
    N = len(tgts)
    print(f"{N} positions (acts now engine scale)", flush=True)

    CHUNK = 32
    chunk_id = np.arange(N) // CHUNK
    rng = np.random.RandomState(42)
    uchunks = np.unique(chunk_id)
    val_chunks = set(rng.choice(uchunks, size=len(uchunks) // 10, replace=False))
    val_mask = np.isin(chunk_id, list(val_chunks))
    tr = np.where(~val_mask)[0]
    va = np.where(val_mask)[0]
    print(f"train {len(tr)}  val {len(va)}", flush=True)

    ev_mean = torch.tensor(evals[tr].mean(0), dtype=torch.float32)
    ev_std = torch.tensor(evals[tr].std(0), dtype=torch.float32).clamp(min=1.0)
    a_tr = acts[tr].reshape(len(tr), -1).astype(np.float32)
    a_mean = torch.tensor(a_tr.mean(0))
    a_std = torch.tensor(a_tr.std(0)).clamp(min=0.05)
    del a_tr

    model = DenseV4(ev_mean, ev_std, a_mean, a_std).to(dev)
    start_ep = 0
    import glob as _g
    if os.path.exists(f"{CKPT_DIR}/last.pt"):
        model.load_state_dict(torch.load(f"{CKPT_DIR}/last.pt", map_location=dev, weights_only=False))
        eps = [int(f.split("ep")[1].split(".")[0]) for f in _g.glob(f"{CKPT_DIR}/ep*.pt")]
        start_ep = (max(eps) + 1) if eps else 1
        print(f"RESUMED from last.pt at ep{start_ep}", flush=True)
    Y = torch.tensor(tgts, dtype=torch.float32)
    X_ev = torch.tensor(evals, dtype=torch.float32)
    X_hce = torch.tensor(hces, dtype=torch.float32)
    oh = np.zeros((N, E_N), dtype=np.float32)
    oh[np.arange(N), doms] = 1.0
    X_oh = torch.tensor(oh)
    X_act = torch.from_numpy(acts)                              # int16-range float
    X_ft = torch.tensor(ft.astype(np.int64))

    opt_emb = torch.optim.SparseAdam([model.emb.weight], lr=5e-3)
    opt = torch.optim.AdamW([
        {"params": model.lin_ev.parameters(), "lr": 3e-4},
        {"params": model.tail.parameters(), "lr": 3e-4},
    ], weight_decay=1e-5)

    BS = 2048
    MAX_EPOCHS = 2000
    STOP_PATIENCE = 10
    DECAY_PATIENCE = 12
    best, best_ep, stale = 1e9, -1, 0
    for ep in range(start_ep, MAX_EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        tl, nb = 0.0, 0
        for i in range(0, len(perm), BS):
            idx = torch.tensor(perm[i:i+BS])
            pred = model(X_ev[idx].to(dev), X_act[idx].to(dev), X_hce[idx].to(dev),
                         X_oh[idx].to(dev), X_ft[idx].to(dev))
            loss = ((wp(pred) - wp(Y[idx].to(dev))) ** 2).mean()
            opt_emb.zero_grad(); opt.zero_grad(); loss.backward(); opt_emb.step(); opt.step()
            tl += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vl = 0.0; nbv = 0
            for i in range(0, len(va), BS):
                idx = torch.tensor(va[i:i+BS])
                pred = model(X_ev[idx].to(dev), X_act[idx].to(dev), X_hce[idx].to(dev),
                             X_oh[idx].to(dev), X_ft[idx].to(dev))
                vl += ((wp(pred) - wp(Y[idx].to(dev))) ** 2).mean().item(); nbv += 1
            vl /= max(nbv, 1)
        torch.save(model.state_dict(), f"{CKPT_DIR}/last.pt")
        if ep % 2 == 1:
            torch.save(model.state_dict(), f"{CKPT_DIR}/ep{ep}.pt")
        if vl < best - 1e-7:
            best, best_ep, stale = vl, ep, 0
            torch.save(model.state_dict(), f"{CKPT_DIR}/best.pt")
        else:
            stale += 1
        if ep % 5 == 0 or ep == MAX_EPOCHS - 1:
            print(f"ep{ep}: train {tl/max(nb,1):.6f}  val {vl:.6f}  best {best:.6f}@{best_ep}", flush=True)
        if stale and stale % DECAY_PATIENCE == 0:
            for o in (opt_emb, opt):
                for gparam in o.param_groups:
                    gparam["lr"] = max(gparam["lr"] * 0.5, 1e-5)
            print(f"ep{ep}: lr decayed (stale {stale})", flush=True)
        if stale >= STOP_PATIENCE:
            print(f"converged: best {best:.6f} at ep{best_ep}; stopping at ep{ep}", flush=True)
            break
    print("TRAINING_DONE", flush=True)

if __name__ == "__main__":
    main()
