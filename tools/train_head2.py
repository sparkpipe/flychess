"""STACKED HEAD trainer v2 — loads precomputed HCE/domain directly,
streams activations from mmap. No FEN parsing, no python-chess in the loop."""
import sys, os, glob, random, struct
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, "/home/spec/chess-lab/tools")

R = "/mnt/cold-raid6/chess-audit"
CACHE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else R + "/stack_cache"
EXPERTS_N = 14
DOMAINS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3","nvb",
           "nvr","bvr","rv2m","qvmat","oppb","dvoretsky","exchanges","none"]
D2I = {d:i for i,d in enumerate(DOMAINS)}

def main():
    smoke = "--smoke" in sys.argv
    torch.manual_seed(42)
    rng = random.Random(42)

    calib = np.load(CACHE + "/calibration.npz")
    scale, zero, D = calib["scale"], calib["zero"], int(calib["D"])
    E_n = EXPERTS_N

    # collect bins — just mmap pointers + targets + precomputed HCE
    bins = []
    for cf in sorted(glob.glob(CACHE + "/*.uint8.npy")):
        name = os.path.basename(cf)[:-10]
        feats = np.load(cf, mmap_mode="r")
        tgts = np.load(CACHE + "/" + name + ".targets.npy")
        hce_f = CACHE + "/" + name + ".hce.npz"
        if not os.path.exists(hce_f):
            print("MISSING HCE:", name, "— skipping")
            continue
        hd = np.load(hce_f)
        n = min(feats.shape[0], len(tgts), hd["hce"].shape[0])
        if smoke:
            n = min(n, 8000)
        bins.append({"name": name, "feats": feats[:n], "tgts": tgts[:n],
                     "hce": hd["hce"][:n], "dom": hd["domain"][:n], "n": n})
    total = sum(b["n"] for b in bins)
    HCE_D = bins[0]["hce"].shape[1]
    print("dataset: %d pos, %d experts, D=%d, HCE=%d, %d bins"
          % (total, E_n, D, HCE_D, len(bins)), flush=True)

    # VECTORIZED feature assembly (no Python loops)
    all_hce = np.concatenate([b["hce"][:b["n"]] for b in bins])
    all_dom = np.concatenate([b["dom"][:b["n"]] for b in bins])
    all_y   = np.concatenate([b["tgts"][:b["n"]] for b in bins])
    # build (bin_idx, local_idx) as numpy arrays
    bin_of = np.concatenate([np.full(b["n"], bi, dtype=np.int32)
                             for bi, b in enumerate(bins)])
    loc_of = np.concatenate([np.arange(b["n"], dtype=np.int32)
                             for b in bins])
    X_hce = torch.from_numpy(all_hce.astype(np.float32))
    X_dom = torch.from_numpy(all_dom.astype(np.int64))
    Y = torch.from_numpy(all_y.astype(np.float32))
    print("features loaded (vectorized): hce=%s dom=%s y=%s" %
          (X_hce.shape, X_dom.shape, Y.shape), flush=True)

    # redefine fetch to use numpy index arrays
    bin_mmaps = [b["feats"] for b in bins]
    def fetch_np(indices):
        """Vectorized fetch using numpy fancy indexing."""
        bi_arr = bin_of[indices]
        lo_arr = loc_of[indices]
        acts = np.zeros((len(indices), E_n, D), dtype=np.float32)
        # group by bin for efficient mmap reads
        for b in np.unique(bi_arr):
            mask = bi_arr == b
            acts[mask] = bin_mmaps[b][lo_arr[mask]].astype(np.float32)
        return acts

    # pad calibration to match actual expert count in features
    actual_E = bins[0]["feats"].shape[1] if bins else E_n
    if len(scale) < actual_E:
        scale = np.pad(scale, (0, actual_E - len(scale)), constant_values=1.0)
        zero = np.pad(zero, (0, actual_E - len(zero)), constant_values=0.0)
    sc = torch.tensor(scale[:actual_E], dtype=torch.float32)[None, :, None]
    ze = torch.tensor(zero[:actual_E], dtype=torch.float32)[None, :, None]
    E_n = actual_E

    def fetch(indices):
        acts = fetch_np(np.array(indices))
        t = torch.from_numpy(acts)
        t = (t / 255.0) * (sc * 255.0) + ze
        return t / (t.abs().mean() + 1e-9)

    perm = np.random.RandomState(42).permutation(total)
    n_val = max(1000, total // 20)
    val_idx, tr_idx = perm[:n_val], perm[n_val:].copy()
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
            oh = torch.zeros(act.shape[0], E_n)
            oh.scatter_(1, dom[:, None], 1.0)
            x = torch.cat([gated.flatten(1), hce, oh], dim=1)
            return self.lin(x).squeeze(-1)

    head = Head()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-6)

    def wp(cp):
        return 1.0 / (1.0 + torch.exp(-cp / 361.0))

    def loss_fn(p, t):
        return ((wp(p) - wp(t)) ** 2).mean()

    best_val = 1e9
    EPOCHS = 3 if smoke else 30
    BS = 4096
    for ep in range(EPOCHS):
        head.train()
        np.random.shuffle(tr_idx)
        tl, nb = 0, 0
        for i in range(0, len(tr_idx), BS):
            idx = tr_idx[i:i + BS]
            acts = fetch(idx)
            pred = head(acts, X_hce[idx], X_dom[idx])
            loss = loss_fn(pred, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        head.eval()
        vl, vps, vts = [], [], []
        with torch.no_grad():
            for i in range(0, len(val_idx), BS):
                idx = val_idx[i:i + BS]
                acts = fetch(idx)
                vp = head(acts, X_hce[idx], X_dom[idx])
                vl.append(loss_fn(vp, Y[idx]).item())
                vps.extend(vp.tolist()); vts.extend(Y[idx].tolist())
        vloss = sum(vl) / len(vl)
        corr = np.corrcoef(vps, vts)[0, 1] if len(vps) > 1 else 0
        print("epoch %d: train=%.6f val=%.6f corr=%.4f"
              % (ep, tl / max(nb, 1), vloss, corr), flush=True)
        if vloss < best_val:
            best_val = vloss
            torch.save(head.state_dict(), R + "/head_v1.pt")

    g = head.gates.detach().numpy()
    print("\ngates:")
    for di, dn in enumerate(DOMAINS[:-1]):
        print("  %-14s %s" % (dn, " ".join("%.2f" % x for x in g[di])))
    print("DONE best_val=%.6f" % best_val)

    # also save in the binary format the engine reads
    save_engine_format(head, R + "/head_v1.stkh", E_n, D, HCE_D, N_DOM)

def save_engine_format(head, path, E, D, H, ND):
    """Write the binary format stack_head.h expects."""
    import struct
    state = head.state_dict()
    gates = state["gates"].numpy()  # (ND, E)
    lw = state["lin.weight"].numpy().flatten()  # (D*E + H + E,)
    lb = state["lin.bias"].numpy().flatten()  # (1,)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIII", 0x53544B48, 1, E, D, H))
        f.write(struct.pack("<I", ND))
        f.write(gates.astype(np.float32).tobytes())
        f.write(lw.astype(np.float32).tobytes())
        f.write(lb.astype(np.float32).tobytes())
        # normalization: scale=1, offset=0 for now (calibration applied in fetch)
        f.write(np.ones(E, dtype=np.float32).tobytes())
        f.write(np.zeros(E, dtype=np.float32).tobytes())
    print("engine format saved:", path)

if __name__ == "__main__":
    main()
