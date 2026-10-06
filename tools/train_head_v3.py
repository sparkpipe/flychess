"""STACKED HEAD v3 — expert scalar evals + routed expert's internal activations
+ similarity-scaled gates.

Architecture (operator design):
  - 13 scalar evals (each expert's FULL evaluation)
  - 1024 activations from the ROUTED expert only (subtle info the scalar misses)
  - 36 HCE features
  - 13 routing one-hot
  - Per-domain gates: 1.0 for routed expert, similarity score (0.1-1.0) for others
  - Linear head (can learn to squelch activations to 0 if they don't help)
"""
import sys, os, glob, struct, time
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")

R = "/mnt/cold-raid6/chess-audit"
DOMAINS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3","nvb",
           "nvr","bvr","rv2m","qvmat","oppb","dvoretsky","exchanges","none"]
D2I = {d:i for i,d in enumerate(DOMAINS)}
EXPERTS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3",
           "nvb","nvr","bvr","rv2m","qvmat","oppb",
           "dvoretsky","exchanges","tactics"]
E_N = len(EXPERTS)
HCE_D = 36
ACT_D = 1024
N_DOM = len(DOMAINS) - 1

# Domain taxonomy for similarity-scaled gates
FAMILY = {"balanced_l0":"balanced","balanced_l1":"balanced",
          "balanced_l2":"balanced","balanced_l3":"balanced",
          "nvb":"confrontation","nvr":"confrontation","bvr":"confrontation",
          "rv2m":"confrontation","qvmat":"confrontation",
          "oppb":"confrontation","dvoretsky":"endgame",
          "exchanges":"transitional","tactics":"tactical"}
PHASE = {"balanced":"mid","confrontation":"mid","endgame":"end",
         "transitional":"mid","tactical":"any"}
ADJACENT = {("balanced_l0","balanced_l1"),("balanced_l1","balanced_l2"),
            ("balanced_l2","balanced_l3"),
            ("nvb","nvr"),("nvb","bvr"),("nvr","bvr"),("nvr","rv2m"),
            ("bvr","rv2m"),("rv2m","qvmat")}

def expert_similarity(e1, e2):
    """Similarity 0.1-1.0 for gate initialization."""
    if e1 == e2:
        return 1.0
    f1, f2 = FAMILY.get(e1,""), FAMILY.get(e2,"")
    s = 0.1  # base
    if f1 == f2:
        s += 0.3  # same family
        if (e1, e2) in ADJACENT or (e2, e1) in ADJACENT:
            s += 0.2  # adjacent within family
    if PHASE.get(f1,"") == PHASE.get(f2,"") and PHASE.get(f1,"") != "":
        s += 0.2  # same phase
    # tactics expert is moderately relevant everywhere
    if e1 == "tactics" or e2 == "tactics":
        s = max(s, 0.3)
    return min(s, 1.0)

def build_gate_init():
    """gates[domain][expert] = 1.0 for routed, similarity for others."""
    g = torch.zeros(N_DOM, E_N)
    for d_idx, dom in enumerate(DOMAINS[:-1]):  # skip "none"
        routed = dom if dom in EXPERTS else EXPERTS[0]
        for e_idx, exp in enumerate(EXPERTS):
            g[d_idx, e_idx] = expert_similarity(routed, exp)
    return g


def main():
    smoke = "--smoke" in sys.argv
    eval_cache = "/srv/workspace/flychess/eval_cache"
    act_cache = "/srv/workspace/flychess/cache"
    out_name = "head_v3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Load: scalar evals, routed expert's activations, HCE, domain, targets
    X_eval_l, X_act_l, X_hce_l, X_dom_l, Y_l = [], [], [], [], []
    total = 0
    for ef in sorted(glob.glob(eval_cache + "/*.evals.npy")):
        name = os.path.basename(ef)[:-10]
        evals = np.load(ef)
        hce_f = act_cache + "/" + name + ".hce.npz"
        tgts_f = act_cache + "/" + name + ".targets.npy"
        if not os.path.exists(hce_f):
            continue
        hd = np.load(hce_f)
        tgts = np.load(tgts_f)
        n = min(evals.shape[0], tgts.shape[0], hd["hce"].shape[0])
        if smoke: n = min(n, 3000)
        else: n = min(n, 25000)  # cap for VRAM
        doms = hd["domain"][:n]
        X_eval_l.append(evals[:n])
        X_hce_l.append(hd["hce"][:n])
        X_dom_l.append(doms)
        Y_l.append(tgts[:n])
        total += n
        print(f"  {name}: {n}", flush=True)

    X_eval = torch.tensor(np.concatenate(X_eval_l), dtype=torch.float32)
    X_hce = torch.tensor(np.concatenate(X_hce_l), dtype=torch.float32)
    X_dom = torch.tensor(np.concatenate(X_dom_l), dtype=torch.long)
    Y = torch.tensor(np.concatenate(Y_l), dtype=torch.float32)
    del X_eval_l, X_hce_l, X_dom_l, Y_l

    # Load ROUTED expert's activations (1024 dims per position)
    print("loading routed expert activations...", flush=True)
    X_act = torch.zeros(total, ACT_D, dtype=torch.float16)
    offset = 0
    for ef in sorted(glob.glob(eval_cache + "/*.evals.npy")):
        name = os.path.basename(ef)[:-10]
        n_done = min(25000, np.load(ef).shape[0] if not smoke else 3000)
        n_done = min(n_done, total - offset)
        if n_done <= 0: break
        doms_np = X_dom[offset:offset+n_done].numpy()
        # load activation cache for this bin
        act_f = act_cache + "/" + name + ".uint8.npy"
        if not os.path.exists(act_f):
            offset += n_done; continue
        acts = np.load(act_f, mmap_mode="r")[:n_done]  # (N, E_n, D)
        for i in range(n_done):
            # routed expert index = domain index (identity mapping for 0-11)
            d = min(doms_np[i], 11)  # clamp "none" to balanced_l0
            X_act[offset + i] = torch.from_numpy(
                acts[i, d].astype(np.float16))
        offset += n_done
        print(f"  acts {name}: {n_done}", flush=True)

    print(f"total: {total} pos", flush=True)
    # to GPU
    X_eval = X_eval.to(device)
    X_act = X_act.to(device)
    X_hce = X_hce.to(device)
    X_dom = X_dom.to(device)
    Y = Y.to(device)

    # normalize evals
    eval_mean = X_eval.mean(dim=0, keepdim=True)
    eval_std = X_eval.std(dim=0, keepdim=True).clamp(min=1.0)
    X_eval = (X_eval - eval_mean) / eval_std
    # normalize activations
    act_mean = X_act.float().mean(dim=0, keepdim=True)
    act_std = X_act.float().std(dim=0, keepdim=True).clamp(min=0.01)
    X_act = ((X_act.float() - act_mean) / act_std).half()

    INPUT_D = E_N + ACT_D + HCE_D + E_N  # 13 + 1024 + 36 + 13 = 1086

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            gate_init = build_gate_init()
            self.gates = nn.Parameter(gate_init)
            # separate weights for eval path and act path
            self.lin_eval = nn.Linear(E_N + HCE_D + E_N, 1, bias=False)
            self.lin_act = nn.Linear(ACT_D, 1, bias=False)
            self.bias = nn.Parameter(torch.zeros(1))
            # init: eval path gets routed weight, act path starts at 0
            nn.init.zeros_(self.lin_eval.weight)
            nn.init.zeros_(self.lin_act.weight)
            with torch.no_grad():
                # eval path: identity for the routed expert's eval
                for e in range(E_N):
                    self.lin_eval.weight[0, e] = 1.0  # each expert's eval passes at unity
        def forward(self, ev, act, hce, dom):
            g = self.gates[dom]
            gated_ev = ev * g
            oh = torch.zeros(ev.shape[0], E_N, device=ev.device)
            oh.scatter_(1, dom[:, None], 1.0)
            eval_part = self.lin_eval(torch.cat([gated_ev, hce, oh], dim=1))
            act_part = self.lin_act(act.float())
            return (eval_part + act_part + self.bias).squeeze(-1)

    head = Head().to(device)
    opt = torch.optim.AdamW([
        {"params": [head.gates], "lr": 1e-4},
        {"params": head.lin_eval.parameters(), "lr": 3e-4},
        {"params": head.lin_act.parameters(), "lr": 1e-4},
        {"params": [head.bias], "lr": 1e-3},
    ], weight_decay=1e-5)

    ckpt_path = f"/srv/workspace/flychess/logs/{out_name}_ckpt.pt"
    start_ep, best_val = 0, 1e9
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        head.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        best_val = ck["best_val"]; start_ep = ck["epoch"] + 1
        print(f"RESUMED epoch {start_ep}", flush=True)

    def wp(cp): return 1.0 / (1.0 + torch.exp(-cp / 361.0))
    def loss_fn(p, t): return ((wp(p) - wp(t)) ** 2).mean()

    perm = np.random.RandomState(42).permutation(total)
    n_val = max(1000, total // 20)
    val_idx = torch.from_numpy(perm[:n_val]).long().to(device)
    tr_idx = torch.from_numpy(perm[n_val:]).long().to(device)

    EPOCHS = 5 if smoke else 500
    BS = 8192
    for ep in range(start_ep, EPOCHS):
        head.train()
        shuffle = tr_idx[torch.randperm(len(tr_idx), device=device)]
        tl, nb = 0.0, 0
        for i in range(0, len(shuffle), BS):
            idx = shuffle[i:i+BS]
            pred = head(X_eval[idx], X_act[idx], X_hce[idx], X_dom[idx])
            loss = loss_fn(pred, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        head.eval()
        with torch.no_grad():
            vl, vps, vts = [], [], []
            for i in range(0, len(val_idx), BS):
                idx = val_idx[i:i+BS]
                vp = head(X_eval[idx], X_act[idx], X_hce[idx], X_dom[idx])
                vl.append(loss_fn(vp, Y[idx]).item())
                vps.extend(vp.cpu().tolist()); vts.extend(Y[idx].cpu().tolist())
        vloss = sum(vl) / len(vl)
        corr = np.corrcoef(vps, vts)[0, 1] if len(vps) > 1 else 0
        print(f"epoch {ep}: train={tl/max(nb,1):.6f} val={vloss:.6f} corr={corr:.4f}", flush=True)
        if vloss < best_val:
            best_val = vloss
            torch.save(head.state_dict(), f"{R}/{out_name}.pt")
        torch.save({"model": head.state_dict(), "optimizer": opt.state_dict(),
                    "best_val": best_val, "epoch": ep,
                    "eval_mean": eval_mean.cpu(), "eval_std": eval_std.cpu(),
                    "act_mean": act_mean.cpu(), "act_std": act_std.cpu()},
                   ckpt_path)

    # save engine format
    sd = head.state_dict()
    gates_np = sd["gates"].cpu().numpy()
    lw_eval = sd["lin_eval.weight"].cpu().numpy().flatten()
    lw_act = sd["lin_act.weight"].cpu().numpy().flatten()
    bias = sd["bias"].cpu().numpy().flatten()
    em = eval_mean.cpu().numpy().flatten()
    es = eval_std.cpu().numpy().flatten()
    am = act_mean.cpu().numpy().flatten()
    ast = act_std.cpu().numpy().flatten()
    with open(f"{R}/{out_name}.evh3", "wb") as f:
        f.write(struct.pack("<IIII", 0x45564C33, 1, E_N, ACT_D))
        f.write(struct.pack("<II", HCE_D, N_DOM))
        f.write(gates_np.astype(np.float32).tobytes())
        f.write(lw_eval.astype(np.float32).tobytes())
        f.write(lw_act.astype(np.float32).tobytes())
        f.write(bias.astype(np.float32).tobytes())
        f.write(em.astype(np.float32).tobytes())
        f.write(es.astype(np.float32).tobytes())
        f.write(am.astype(np.float32).tobytes())
        f.write(ast.astype(np.float32).tobytes())
    print(f"DONE best_val={best_val:.6f}, engine: {R}/{out_name}.evh3", flush=True)
    print(f"gates:\n{gates_np}")


if __name__ == "__main__":
    main()
