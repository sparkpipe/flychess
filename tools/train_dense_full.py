"""FULL-CORPUS supervised training — 29.7M positions, depth-12 targets.

Reads sidecars_full/*_p*.npz parts (evals, acts engine-scale, ft, cps).
Same DenseV4 architecture; train-until-stops with lr decay; checkpoints.
"""
import sys, os, glob, time, math
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import torch, torch.nn as nn

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
SF = f"{R}/sidecars_full"
E_N, HCE_D, ACT_D, FT_D = 13, 36, 1024, 86896
H1, H2 = 64, 32
CKPT_DIR = f"{SP}/dense_full_ckpts"
MAXF = 160

# HCE recomputed would be expensive over 29.7M; use eval-order domain one-hot
# from the bin name (each aug bin = one expert domain) + material from evals.
# Domain per bin: the expert the bin belongs to.
BIN2DOM = {}
DOMS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3","nvb",
        "nvr","bvr","rv2m","qvmat","oppb","dvoretsky","exchanges","tb"]
for i, d in enumerate(DOMS):
    BIN2DOM[d] = i
for extra in ["tactics", "aug_tactics", "gambit"]:
    BIN2DOM[extra] = 12

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
        raw = (self.emb(ftidx.clamp(min=0)) * mask).sum(dim=1)
        feats = torch.cat([x_act.view(len(ev), -1), raw, hce, oh], dim=1)
        return self.tail(feats).squeeze(-1) + self.lin_ev(x_ev).squeeze(-1)

def hce_from_evals(evals):
    """Cheap HCE proxy from eval matrix columns is wrong; compute material
    from the eval columns themselves is not possible. Instead: zeros HCE —
    the raw-board embedding subsumes material/structure information."""
    return np.zeros((evals.shape[0], HCE_D), dtype=np.float32)

class PartStore:
    """Memory-mapped-ish store over the npz parts, one array per field."""
    def __init__(self, files):
        self.files = sorted(files)
        self.index = []   # (file_idx, row_offset, count)
        for fi, f in enumerate(self.files):
            import zipfile
            with zipfile.ZipFile(f) as z:
                info = z.getinfo("evals.npy")
                n = (info.file_size - 128) // (13 * 4)
            self.index.append([fi, 0, n])
        self.total = sum(ix[2] for ix in self.index)
    def __len__(self):
        return self.total

def load_batch_data(files, max_rows=None):
    """Load all parts into RAM as uint8/int16/int32 arrays (acts are the bulk)."""
    evals_l, acts_l, ft_l, cnt_l, cps_l = [], [], [], [], []
    total = 0
    for f in sorted(files):
        d = np.load(f)
        n = d["evals"].shape[0]
        if max_rows and total + n > max_rows:
            n = max_rows - total
        evals_l.append(d["evals"][:n])
        acts_l.append(d["acts"][:n])
        ft_l.append(d["ft"][:n])
        cnt_l.append(d["ftcnt"][:n])
        cps_l.append(d["cps"][:n])
        total += n
        del d
        if max_rows and total >= max_rows:
            break
    return (np.concatenate(evals_l), np.concatenate(acts_l),
            np.concatenate(ft_l), np.concatenate(cnt_l), np.concatenate(cps_l))

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    os.makedirs(CKPT_DIR, exist_ok=True)
    parts = sorted(glob.glob(f"{SF}/*_p*.npz"))
    print(f"{len(parts)} parts", flush=True)
    evals, acts, ft, cnt, cps = load_batch_data(parts)
    N = len(cps)
    print(f"loaded {N:,} positions", flush=True)

    # bin-domain one-hots from part filenames
    doms = np.zeros(N, dtype=np.int64)
    off = 0
    for f in parts:
        d_name = os.path.basename(f).split("_p")[0].replace("aug_", "")
        dom = next((BIN2DOM[k] for k in BIN2DOM if d_name.startswith(k)), 12)
        n = np.load(f, mmap_mode=None)["evals"].shape[0]
        end = min(off + n, N)
        doms[off:end] = dom
        off = end
        if off >= N:
            break
    oh_all = np.zeros((N, E_N), dtype=np.float32)
    oh_all[np.arange(N), doms] = 1.0

    # chunked split (32-record chunks from same games)
    CHUNK = 32
    chunk_id = np.arange(N) // CHUNK
    rng = np.random.RandomState(42)
    uchunks = np.unique(chunk_id)
    val_chunks = set(rng.choice(uchunks, size=len(uchunks) // 10, replace=False))
    val_mask = np.isin(chunk_id, list(val_chunks))
    tr = np.where(~val_mask)[0]
    va = np.where(val_mask)[0]
    print(f"train {len(tr):,}  val {len(va):,}", flush=True)

    # normalization stats from a train sample (1M rows for speed)
    samp = tr[:: max(1, len(tr) // 1_000_000)]
    ev_mean = torch.tensor(evals[samp].mean(0), dtype=torch.float32)
    ev_std = torch.tensor(evals[samp].std(0), dtype=torch.float32).clamp(min=1.0)
    a_s = acts[samp].reshape(len(samp), -1).astype(np.float32)
    a_mean = torch.tensor(a_s.mean(0))
    a_std = torch.tensor(a_s.std(0)).clamp(min=0.05)
    del a_s, samp

    model = DenseV4(ev_mean, ev_std, a_mean, a_std).to(dev)
    # optionally warm-start from the 400k run
    warm = f"{SP}/dense_v4_ckpts3/best.pt"
    if os.path.exists(warm) and "--cold" not in sys.argv:
        try:
            sd = torch.load(warm, map_location=dev, weights_only=False)
            model.load_state_dict(sd, strict=False)
            print("warm-started from 400k best", flush=True)
        except Exception as e:
            print("warm start skipped:", e, flush=True)

    Y = torch.tensor(cps.astype(np.float32))
    X_ev = torch.tensor(evals)
    X_act = torch.from_numpy(acts)
    X_ft = torch.tensor(ft.astype(np.int64))
    X_oh = torch.tensor(oh_all)
    hces = hce_from_evals(evals)
    X_hce = torch.tensor(hces)
    del evals, acts, ft, oh_all, hces, cps

    opt_emb = torch.optim.SparseAdam([model.emb.weight], lr=5e-3)
    opt = torch.optim.AdamW([p for n, p in model.named_parameters()
                             if not n.startswith("emb.")], lr=3e-4, weight_decay=1e-5)

    BS = 4096
    MAX_EPOCHS = 2000
    STOP_PATIENCE = 10
    DECAY_PATIENCE = 12
    best, best_ep, stale = 1e9, -1, 0
    start_ep = 0
    import glob as _g
    if os.path.exists(f"{CKPT_DIR}/last.pt"):
        model.load_state_dict(torch.load(f"{CKPT_DIR}/last.pt", map_location=dev, weights_only=False))
        eps = [int(f.split("ep")[1].split(".")[0]) for f in _g.glob(f"{CKPT_DIR}/ep*.pt")]
        start_ep = (max(eps) + 1) if eps else 1
        print(f"RESUMED at ep{start_ep}", flush=True)

    for ep in range(start_ep, MAX_EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        tl, nb = 0.0, 0
        for i in range(0, len(perm), BS):
            idx = torch.tensor(perm[i:i+BS])
            pred = model(X_ev[idx].to(dev), X_act[idx].to(dev), X_hce[idx].to(dev),
                         X_oh[idx].to(dev), X_ft[idx].to(dev))
            loss = ((wp(pred) - wp(Y[idx].to(dev))) ** 2).mean()
            opt_emb.zero_grad(); opt.zero_grad()
            loss.backward()
            opt_emb.step(); opt.step()
            tl += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vl = 0.0; nbv = 0
            for i in range(0, len(va), BS):
                idx = va[i:i+BS]
                tidx = torch.tensor(idx)
                pred = model(X_ev[tidx].to(dev), X_act[tidx].to(dev), X_hce[tidx].to(dev),
                             X_oh[tidx].to(dev), X_ft[tidx].to(dev))
                vl += ((wp(pred) - wp(Y[tidx].to(dev))) ** 2).mean().item(); nbv += 1
            vl /= max(nbv, 1)
        torch.save(model.state_dict(), f"{CKPT_DIR}/last.pt")
        if ep % 2 == 1:
            torch.save(model.state_dict(), f"{CKPT_DIR}/ep{ep}.pt")
        if vl < best - 1e-7:
            best, best_ep, stale = vl, ep, 0
            torch.save(model.state_dict(), f"{CKPT_DIR}/best.pt")
        else:
            stale += 1
        if stale and stale % DECAY_PATIENCE == 0:
            for o in (opt_emb, opt):
                for g in o.param_groups:
                    g["lr"] = max(g["lr"] * 0.5, 1e-5)
            print(f"ep{ep}: lr decayed (stale {stale})", flush=True)
        if ep % 5 == 0 or ep == MAX_EPOCHS - 1:
            print(f"ep{ep}: train {tl/max(nb,1):.6f}  val {vl:.6f}  best {best:.6f}@{best_ep}", flush=True)
        if ep % 10 == 5 and os.path.exists(f"{SP}/gate_cache/features.npz"):
            os.system(f"python3 /srv/workspace/flychess/src/chess-lab/tools/accuracy_gate.py {CKPT_DIR}/last.pt full-ep{ep} >> /mnt/cold-raid6/chess-audit/selfplay_rl/gate_runs.log 2>&1")
        if stale >= STOP_PATIENCE:
            print(f"converged: best {best:.6f} at ep{best_ep}; stopping at ep{ep}", flush=True)
            break
    print("TRAINING_DONE", flush=True)

if __name__ == "__main__":
    main()
