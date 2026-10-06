"""STACKED HEAD trainer — GPU-accelerated. Pre-loads all activations to VRAM,
making epochs take seconds instead of minutes."""
import sys, os, glob, struct, random, time
import numpy as np
import torch, torch.nn as nn

R = "/mnt/cold-raid6/chess-audit"
DOMAINS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3","nvb",
           "nvr","bvr","rv2m","qvmat","oppb","dvoretsky","exchanges","none"]
D2I = {d:i for i,d in enumerate(DOMAINS)}

def main():
    smoke = "--smoke" in sys.argv
    cache = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/srv/workspace/flychess/cache"
    out_name = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "head_v1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    print(f"device={device} cache={cache} out={out_name}", flush=True)

    calib = np.load(cache + "/calibration.npz")
    scale, zero, D = calib["scale"], calib["zero"], int(calib["D"])

    # Load all bins: activations eager to GPU, HCE/dom/targets as tensors
    all_acts, all_hce, all_dom, all_y = [], [], [], []
    total = 0
    MAX = 15000  # per bin
    for cf in sorted(glob.glob(cache + "/*.uint8.npy")):
        name = os.path.basename(cf)[:-10]
        tgts = np.load(cache + "/" + name + ".targets.npy")
        hce_f = cache + "/" + name + ".hce.npz"
        if not os.path.exists(hce_f):
            continue
        hd = np.load(hce_f)
        feats = np.load(cf, mmap_mode="r")
        n = min(feats.shape[0], len(tgts), hd["hce"].shape[0])
        if smoke: n = min(n, 3000)
        else: n = min(n, MAX)
        acts = torch.from_numpy(feats[:n].astype(np.float16)).to(device)
        all_acts.append(acts)
        all_hce.append(torch.from_numpy(hd["hce"][:n].astype(np.float32)))
        all_dom.append(torch.from_numpy(hd["domain"][:n].astype(np.int64)))
        all_y.append(torch.from_numpy(tgts[:n].astype(np.float32)))
        total += n
        print(f"  {name}: {n}", flush=True)
    acts_gpu = torch.cat(all_acts).to(device)
    X_hce = torch.cat(all_hce)
    X_dom = torch.cat(all_dom)
    Y = torch.cat(all_y)
    del all_acts, all_hce, all_dom, all_y
    E_n = acts_gpu.shape[1]
    HCE_D = X_hce.shape[1]
    print(f"total: {total} pos, acts={acts_gpu.shape}, hce={X_hce.shape}", flush=True)

    # calibration constants
    # Pre-calibrate and normalize activations on GPU (one-time)
    # No pre-calibration: the linear layer learns the scale directly.
    # This eliminates the train/engine calibration mismatch completely.
    X_hce = X_hce.to(device)
    X_dom = X_dom.to(device)
    Y = Y.to(device)

    N_DOM = len(DOMAINS) - 1
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.gates = nn.Parameter(torch.ones(N_DOM, E_n))
            self.lin = nn.Linear(D * E_n + HCE_D + E_n, 1)
            nn.init.zeros_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)
        def forward(self, act, hce, dom):
            g = self.gates[dom]
            gated = act * g[:, :, None]
            oh = torch.zeros(act.shape[0], E_n, device=act.device)
            oh.scatter_(1, dom[:, None], 1.0)
            x = torch.cat([gated.flatten(1), hce, oh], dim=1)
            return self.lin(x).squeeze(-1)

    head = Head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-6)

    # checkpoint/resume
    ckpt_path = f"/srv/workspace/flychess/logs/{out_name}_gpu_ckpt.pt"
    start_ep, best_val = 0, 1e9
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        head.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        best_val = ck["best_val"]; start_ep = ck["epoch"] + 1
        print(f"RESUMED epoch {start_ep} best={best_val:.6f}", flush=True)

    def wp(cp): return 1.0 / (1.0 + torch.exp(-cp / 361.0))
    def loss_fn(p, t): return ((wp(p) - wp(t)) ** 2).mean()

    perm = np.random.RandomState(42).permutation(total)
    n_val = max(1000, total // 20)
    val_idx = torch.from_numpy(perm[:n_val]).long().to(device)
    tr_idx = torch.from_numpy(perm[n_val:]).long().to(device)

    EPOCHS = 3 if smoke else 300
    BS = 16384
    for ep in range(start_ep, EPOCHS):
        head.train()
        shuffle = tr_idx[torch.randperm(len(tr_idx), device=device)]
        tl, nb = 0.0, 0
        for i in range(0, len(shuffle), BS):
            idx = shuffle[i:i+BS]
            pred = head(acts_gpu[idx], X_hce[idx], X_dom[idx])
            loss = loss_fn(pred, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        head.eval()
        with torch.no_grad():
            vl, vps, vts = [], [], []
            for i in range(0, len(val_idx), BS):
                idx = val_idx[i:i+BS]
                vp = head(acts_gpu[idx], X_hce[idx], X_dom[idx])
                vl.append(loss_fn(vp, Y[idx]).item())
                vps.extend(vp.cpu().tolist()); vts.extend(Y[idx].cpu().tolist())
        vloss = sum(vl) / len(vl)
        corr = np.corrcoef(vps, vts)[0, 1] if len(vps) > 1 else 0
        print(f"epoch {ep}: train={tl/max(nb,1):.6f} val={vloss:.6f} corr={corr:.4f}", flush=True)
        if vloss < best_val:
            best_val = vloss
            torch.save(head.state_dict(), f"{R}/{out_name}.pt")
        torch.save({"model": head.state_dict(), "optimizer": opt.state_dict(),
                    "best_val": best_val, "epoch": ep}, ckpt_path)

    # save engine format
    sd = head.state_dict()
    gates_np = sd["gates"].cpu().numpy()
    if gates_np.shape[0] < 13:
        padded = np.ones((13, E_n), dtype=np.float32)
        padded[:gates_np.shape[0]] = gates_np
        gates_np = padded
    lw = sd["lin.weight"].cpu().numpy().flatten()
    lb = sd["lin.bias"].cpu().numpy().flatten()
    sc_p = np.ones(14, dtype=np.float32); ze_p = np.zeros(14, dtype=np.float32)
    sc_p[:len(scale)] = scale; ze_p[:len(zero)] = zero
    with open(f"{R}/{out_name}.stkh", "wb") as f:
        f.write(struct.pack("<IIIII", 0x53544B48, 1, E_n, D, HCE_D))
        f.write(struct.pack("<I", 13))
        f.write(gates_np.astype(np.float32).tobytes())
        f.write(lw.astype(np.float32).tobytes())
        f.write(lb.astype(np.float32).tobytes())
        f.write(sc_p.tobytes()); f.write(ze_p.tobytes())
    print(f"DONE best_val={best_val:.6f}, engine file: {R}/{out_name}.stkh", flush=True)

if __name__ == "__main__":
    main()
